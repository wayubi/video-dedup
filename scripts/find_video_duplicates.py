#!/usr/bin/env python3
"""
Video Duplicate Detector
Identifies duplicate videos based on content (not file hash).
Uses audio fingerprinting as primary method, visual fingerprinting as fallback.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import hashlib
import fcntl
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import audioread
from scipy.ndimage import convolve

VideoFeatures = Dict[str, Any]


def compare_features(f1: VideoFeatures, f2: VideoFeatures, verbose: bool = False) -> Tuple[bool, str]:
    """Compare two videos using precomputed features with multi-stage filtering."""
    v1_name = os.path.basename(f1.get("path", ""))
    v2_name = os.path.basename(f2.get("path", ""))
    dur1, dur2 = f1.get("duration", 0), f2.get("duration", 0)

    if verbose:
        print(f"  Comparing {v1_name} vs {v2_name}")

    if dur1 <= 0 or dur2 <= 0:
        return False, "no_duration"

    # STAGE 0: File hash (instant - identical files)
    hash1 = f1.get("file_hash", "")
    hash2 = f2.get("file_hash", "")
    if hash1 and hash2 and hash1 == hash2:
        if verbose:
            print(f"    Stage 0 (File Hash): MATCH")
        return True, "file_hash"

    # STAGE 1: Duration (fastest)
    duration_diff = abs(dur1 - dur2) / max(dur1, dur2, 1)
    if verbose:
        print(f"    Stage 1 (Duration): diff={duration_diff:.4f}, threshold={LENGTH_TOLERANCE}, result={'PASS' if duration_diff <= LENGTH_TOLERANCE else 'FAIL'}")
    if duration_diff > LENGTH_TOLERANCE:
        return False, "duration_mismatch"

    # STAGE 2: Aspect ratio check
    res1 = f1.get("resolution", (0, 0))
    res2 = f2.get("resolution", (0, 0))
    if res1 != (0, 0) and res2 != (0, 0):
        ratio1 = res1[0] / res1[1] if res1[1] > 0 else 0
        ratio2 = res2[0] / res2[1] if res2[1] > 0 else 0
        ratio_diff = abs(ratio1 - ratio2)
        if verbose:
            print(f"    Stage 2 (Aspect Ratio): ratio1={ratio1:.3f}, ratio2={ratio2:.3f}, diff={ratio_diff:.4f}, threshold=0.05, result={'PASS' if ratio_diff <= 0.05 else 'FAIL'}")
        if ratio_diff > 0.05:
            return False, "aspect_ratio_mismatch"

    # STAGE 3: File size
    size1 = f1.get("file_size", 0)
    size2 = f2.get("file_size", 0)
    if size1 > 0 and size2 > 0:
        size_diff = abs(size1 - size2) / max(size1, size2)
        if verbose:
            print(f"    Stage 3 (File Size): size1={size1}, size2={size2}, diff={size_diff:.4f}, threshold=0.90, result={'PASS' if size_diff <= 0.90 else 'FAIL'}")
        if size_diff > 0.90:
            return False, "size_mismatch"

    # STAGE 4: Audio fingerprint
    if f1.get("has_audio") and f2.get("has_audio"):
        fps1 = f1.get("audio_fingerprints", [])
        fps2 = f2.get("audio_fingerprints", [])

        if verbose:
            has_fps1 = len(fps1) > 0
            has_fps2 = len(fps2) > 0
            print(f"    Stage 4 (Audio): has_audio=true, fps1={has_fps1}, fps2={has_fps2}")

        if fps1 and fps2:
            matches = 0
            for fp1_list in fps1:
                fp1_arr = np.array(fp1_list)
                best_sim = 0.0
                for fp2_list in fps2:
                    fp2_arr = np.array(fp2_list)
                    sim = compare_audio_fingerprints(fp1_arr, fp2_arr)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim > 0.90:
                    matches += 1

            required = max(1, len(fps1) - 1)
            if verbose:
                print(f"      Audio matches: {matches}/{len(fps1)} (required={required})")
            if matches >= required:
                return True, "audio_fingerprint"

    # STAGE 5: Visual hash
    hashes1 = f1.get("visual_hashes", [])
    hashes2 = f2.get("visual_hashes", [])

    if verbose:
        print(f"    Stage 5 (Visual): hashes1={len(hashes1)} frames, hashes2={len(hashes2)} frames")

    if hashes1 and hashes2:
        visual_sim = compare_visual_fingerprints(hashes1, hashes2, MAX_VISUAL_OFFSET)
        if verbose:
            print(f"      Visual similarity: {visual_sim:.4f}, threshold=0.25")
        if visual_sim >= 0.25:
            return True, "visual_fingerprint"

    return False, "no_match"


# ---------------------------------------------------------------------------
# Supported video extensions
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LENGTH_TOLERANCE = 0.15          # 15% duration difference allowed

# Audio sampling — longer windows (20 s) give the cross-correlation enough
# shared content to detect an offset even when one video has a ~6-second intro.
# 3 × 20 s = 60 s of audio analysed per video, spread across the file.
AUDIO_SAMPLE_DURATION = 20       # seconds per audio sample (was 5)
NUM_AUDIO_SAMPLES = 3            # number of samples (was 4)

# Maximum intro / offset length we try to compensate for.
# Cross-correlation will search for alignment within ±this many seconds.
MAX_AUDIO_OFFSET_SECONDS = 30

# Derived constant: fingerprint values produced per second of audio.
# hop_length=256, sr=16000 → 62.5 STFT frames/s
# grouped into windows of 10 frames → 6.25 time windows/s
# 6 frequency bands → 6.25 × 6 = 37.5 fingerprint values/s
_FP_VALUES_PER_SECOND = 37.5

NUM_VISUAL_SAMPLES = 4
SKIP_FIRST_SECONDS = 10
MAX_VISUAL_OFFSET = 3  # For visual offset search (±3 frame positions)
TEMP_DIR = None

# Upscaling Detection Configuration
UPSCALING_CONFIDENCE_THRESHOLD = 0.65
MIN_UPSCALING_ANALYSIS_RESOLUTION = 720


# ---------------------------------------------------------------------------
# Video duration / metadata helpers
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Get all video metadata in a single ffprobe call."""
    result = {"duration": 0.0, "resolution": (0, 0), "has_audio": False}
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=width,height,codec_type',
            '-of', 'json', video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)

        if "format" in data and "duration" in data["format"]:
            result["duration"] = float(data["format"]["duration"])

        if "streams" in data:
            for stream in data["streams"]:
                if stream.get("codec_type") == "video":
                    result["resolution"] = (stream.get("width", 0), stream.get("height", 0))
                elif stream.get("codec_type") == "audio":
                    result["has_audio"] = True
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Audio fingerprinting
# ---------------------------------------------------------------------------

def extract_audio_sample(video_path: str, start_time: float, duration: float) -> Optional[np.ndarray]:
    """Extract audio sample from video at a specific time, resampled to 16 kHz mono."""
    if TEMP_DIR is None:
        return None
    try:
        temp_wav = os.path.join(TEMP_DIR, f"audio_{os.path.basename(video_path)}_{start_time}.wav")
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
            '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            temp_wav
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)

        if not os.path.exists(temp_wav):
            return None

        with audioread.audio_open(temp_wav) as f:
            audio_data = []
            for buf in f:
                audio_data.append(np.frombuffer(buf, dtype=np.int16))
            if audio_data:
                audio_array = np.concatenate(audio_data)
                os.remove(temp_wav)
                return audio_array
        return None
    except Exception:
        return None


def generate_audio_fingerprint(audio_data: np.ndarray) -> np.ndarray:
    """Generate a spectral fingerprint from raw audio samples."""
    audio_data = audio_data.astype(np.float32) / 32768.0

    n_fft = 512
    hop_length = 256

    if len(audio_data) < n_fft:
        audio_data = np.pad(audio_data, (0, n_fft - len(audio_data)))

    frames = []
    for i in range(0, len(audio_data) - n_fft + 1, hop_length):
        frame = audio_data[i:i + n_fft]
        windowed = frame * np.hanning(n_fft)
        fft = np.fft.rfft(windowed)
        frames.append(np.abs(fft))

    if not frames:
        return np.array([])

    spectrogram = np.array(frames).T

    bands = [0, 10, 20, 40, 80, 160, 512]
    fingerprint = []
    time_windows = np.array_split(spectrogram, max(1, spectrogram.shape[1] // 10), axis=1)

    for window in time_windows:
        if window.size == 0:
            continue
        for i in range(len(bands) - 1):
            band_start = bands[i]
            band_end = min(bands[i + 1], window.shape[0])
            if band_end > band_start:
                band_energy = np.sum(window[band_start:band_end, :], axis=0)
                if len(band_energy) > 0:
                    fingerprint.append(band_energy[np.argmax(band_energy)])

    if fingerprint:
        fp = np.array(fingerprint)
        return fp / (np.max(fp) + 1e-10)
    return np.array([])


def has_audio_stream(video_path: str) -> bool:
    """Check if video has an audio stream."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return 'audio' in result.stdout.lower()
    except Exception:
        return False


def compare_audio_fingerprints(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compare two audio fingerprints using sliding cross-correlation.

    Unlike a simple point-wise correlation, this function shifts fp2 relative
    to fp1 across a window of ±MAX_AUDIO_OFFSET_SECONDS and returns the best
    similarity found at any offset.  This means two videos that are identical
    except one has an intro of up to MAX_AUDIO_OFFSET_SECONDS will still be
    detected as duplicates.

    The fingerprint is produced at ~37.5 values/second (see _FP_VALUES_PER_SECOND),
    so a 30-second search window corresponds to ±1125 array positions.
    """
    if len(fp1) == 0 or len(fp2) == 0:
        return 0.0

    std1, std2 = np.std(fp1), np.std(fp2)
    MIN_VAR = 0.1
    if std1 < MIN_VAR or std2 < MIN_VAR:
        return 0.0

    # Zero-mean, unit-variance normalisation
    fp1 = (fp1 - np.mean(fp1)) / (std1 + 1e-10)
    fp2 = (fp2 - np.mean(fp2)) / (std2 + 1e-10)

    # Truncate to the same length before cross-correlating
    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]

    # Full cross-correlation — O(n²) but n ≤ ~750 for 20-second windows, fast enough
    correlation = np.correlate(fp1, fp2, mode='full')

    # Restrict the search to ±MAX_AUDIO_OFFSET_SECONDS worth of shift
    max_shift = int(MAX_AUDIO_OFFSET_SECONDS * _FP_VALUES_PER_SECOND)
    center = len(correlation) // 2
    lo = max(0, center - max_shift)
    hi = min(len(correlation), center + max_shift + 1)
    best_raw = float(np.max(correlation[lo:hi]))

    # Normalise: perfect alignment of identical arrays gives correlation = min_len
    similarity = best_raw / (min_len + 1e-10)
    return float(np.clip((similarity + 1) / 2, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Visual fingerprinting
# ---------------------------------------------------------------------------

def extract_visual_samples(video_path: str, duration: float) -> List[Image.Image]:
    """Extract frame samples from video at specific timestamps."""
    if TEMP_DIR is None:
        return []
    images = []
    for point in [0.2, 0.4, 0.6, 0.8][:NUM_VISUAL_SAMPLES]:
        timestamp = duration * point
        try:
            temp_frame = os.path.join(TEMP_DIR, f"frame_{os.path.basename(video_path)}_{point}.jpg")
            cmd = [
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-q:v', '2', temp_frame
            ]
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.exists(temp_frame):
                images.append(Image.open(temp_frame))
                os.remove(temp_frame)
        except Exception:
            pass
    return images


def generate_visual_fingerprint(images: List[Image.Image]) -> List[List[str]]:
    """Generate region-based perceptual hashes per frame (3x3 grid)."""
    hashes = []
    for img in images:
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            region_hashes = []
            for row in range(3):
                y1 = int(h * row / 3)
                y2 = int(h * (row + 1) / 3)
                for col in range(3):
                    x1 = int(w * col / 3)
                    x2 = int(w * (col + 1) / 3)
                    region = img.crop((x1, y1, x2, y2))
                    region_hashes.append(str(imagehash.phash(region)))
            hashes.append(region_hashes)
        except Exception:
            pass
    return hashes


def hamming_distance(hash1: str, hash2: str) -> int:
    """Hamming distance between two hex hashes."""
    if len(hash1) != len(hash2):
        return 1000
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def compare_visual_fingerprints(hashes1: List[List[str]], hashes2: List[List[str]], max_offset: int = 1) -> float:
    """
    Compare visual fingerprints and return similarity (0-1).
    Each frame has 9 regions (3x3 grid). A frame is "good" if at least 5 of 9
    regions match (distance <= 9), allowing for 1-2 regions with watermarks
    or other differences.

    Uses sliding window offset search to handle videos where one has an intro
    that shifts the content. For each frame position, searches within ±max_offset
    positions for the best match.
    """
    if not hashes1 or not hashes2:
        return 0.0

    n1, n2 = len(hashes1), len(hashes2)
    if n1 == 0 or n2 == 0:
        return 0.0

    threshold_match = 9

    def frame_similarity(h1: List[str], h2: List[str]) -> float:
        if not h1 or not h2:
            return 0.0
        region_matches = 0
        for r1, r2 in zip(h1, h2):
            if hamming_distance(r1, r2) <= threshold_match:
                region_matches += 1
        return 1.0 if region_matches >= 5 else 0.0

    best_count = 0

    for i in range(n1):
        regions1 = hashes1[i]
        best_frame_match = 0.0

        lo = max(0, i - max_offset)
        hi = min(n2, i + max_offset + 1)

        for j in range(lo, hi):
            sim = frame_similarity(regions1, hashes2[j])
            if sim > best_frame_match:
                best_frame_match = sim

        if best_frame_match > 0:
            best_count += 1

    return best_count / max(n1, n2)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_videos(
    directory: str,
    include_subfolders: Optional[List[str]] = None,
    exclude_root: bool = False
) -> List[str]:
    """Find all video files in directory."""
    videos = []
    abs_directory = os.path.abspath(directory)
    deduped_folder = os.path.join(abs_directory, "__deduped")

    def is_in_deduped(path: str) -> bool:
        try:
            return (os.path.commonpath([path, deduped_folder]) == deduped_folder or
                    path.startswith(deduped_folder + os.sep))
        except ValueError:
            return False

    if include_subfolders is None:
        if not exclude_root:
            for file in os.listdir(abs_directory):
                file_path = os.path.join(abs_directory, file)
                if os.path.isfile(file_path) and Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(file_path)
    else:
        if len(include_subfolders) == 0:
            for root, _, files in os.walk(abs_directory):
                if is_in_deduped(root):
                    continue
                if exclude_root and root == abs_directory:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(file_path)
        else:
            folders_to_scan = [] if exclude_root else [abs_directory]
            for subfolder in include_subfolders:
                subfolder_path = (os.path.abspath(subfolder) if os.path.isabs(subfolder)
                                  else os.path.abspath(os.path.join(abs_directory, subfolder)))
                if not os.path.exists(subfolder_path):
                    print(f"Warning: Specified subfolder does not exist: {subfolder}")
                    continue
                if not os.path.isdir(subfolder_path):
                    print(f"Warning: Specified path is not a directory: {subfolder}")
                    continue
                if is_in_deduped(subfolder_path):
                    print(f"Warning: Skipping __deduped folder: {subfolder}")
                    continue
                folders_to_scan.append(subfolder_path)

            for folder in folders_to_scan:
                for root, _, files in os.walk(folder):
                    if is_in_deduped(root):
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                            videos.append(file_path)

    return videos


# ---------------------------------------------------------------------------
# Legacy per-pair comparison (used by find_duplicate_groups)
# ---------------------------------------------------------------------------

def is_same_video(video1: Tuple[str, float], video2: Tuple[str, float]) -> Tuple[bool, str]:
    path1, dur1 = video1
    path2, dur2 = video2

    duration_diff = abs(dur1 - dur2) / max(dur1, dur2, 1)
    if duration_diff > LENGTH_TOLERANCE:
        return False, "duration_mismatch"

    audio1 = has_audio_stream(path1)
    audio2 = has_audio_stream(path2)

    if audio1 and audio2:
        samples_match = 0
        total_samples = 0
        min_dur = min(dur1, dur2)
        sample_starts: List[float] = []

        if min_dur > SKIP_FIRST_SECONDS + AUDIO_SAMPLE_DURATION:
            available = min_dur - SKIP_FIRST_SECONDS - AUDIO_SAMPLE_DURATION
            for i in range(NUM_AUDIO_SAMPLES):
                start = SKIP_FIRST_SECONDS + (available * i / max(NUM_AUDIO_SAMPLES - 1, 1))
                sample_starts.append(start)
        else:
            mid = min_dur / 2
            sample_starts = [max(0, mid - AUDIO_SAMPLE_DURATION / 2)]

        for start in sample_starts:
            d1 = extract_audio_sample(path1, start, AUDIO_SAMPLE_DURATION)
            d2 = extract_audio_sample(path2, start, AUDIO_SAMPLE_DURATION)
            if d1 is not None and d2 is not None:
                total_samples += 1
                fp1 = generate_audio_fingerprint(d1)
                fp2 = generate_audio_fingerprint(d2)
                if compare_audio_fingerprints(fp1, fp2) > 0.90:
                    samples_match += 1

        required = max(1, total_samples - 1)
        if total_samples > 0 and samples_match >= required:
            return True, "audio_fingerprint"

    images1 = extract_visual_samples(path1, dur1)
    images2 = extract_visual_samples(path2, dur2)

    if len(images1) >= 2 and len(images2) >= 2:
        hashes1 = generate_visual_fingerprint(images1)
        hashes2 = generate_visual_fingerprint(images2)
        if compare_visual_fingerprints(hashes1, hashes2) >= 0.5:
            return True, "visual_fingerprint"

    return False, "no_match"


# ---------------------------------------------------------------------------
# Duplicate group finding
# ---------------------------------------------------------------------------

def find_duplicate_groups(videos: List[Tuple[str, float]]) -> List[List[str]]:
    n = len(videos)
    if n == 0:
        return []

    duration_groups: Dict[int, List] = defaultdict(list)
    for video in videos:
        bucket = round(video[1] / 30) * 30
        duration_groups[bucket].append(video)

    matches: Dict[str, Set[str]] = defaultdict(set)
    print(f"Analyzing {n} videos for duplicates...")

    for bucket, bucket_videos in duration_groups.items():
        bucket_size = len(bucket_videos)
        if bucket_size < 2:
            continue
        for i in tqdm(range(bucket_size), desc=f"Bucket {bucket}s"):
            for j in range(i + 1, bucket_size):
                is_dup, _ = is_same_video(bucket_videos[i], bucket_videos[j])
                if is_dup:
                    matches[bucket_videos[i][0]].add(bucket_videos[j][0])
                    matches[bucket_videos[j][0]].add(bucket_videos[i][0])

    visited: Set[str] = set()
    groups = []
    for video in videos:
        vp = video[0]
        if vp in visited:
            continue
        group = []
        queue = [vp]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            for neighbor in matches.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(group) > 1:
            groups.append(group)
    return groups


def find_duplicate_groups_with_features(features_list: List[VideoFeatures], verbose: bool = False) -> List[List[str]]:
    n = len(features_list)
    if n == 0:
        return []

    duration_groups: Dict[int, List[VideoFeatures]] = defaultdict(list)
    for f in features_list:
        bucket = round(f.get("duration", 0) / 30) * 30
        duration_groups[bucket].append(f)

    matches: Dict[str, Set[str]] = defaultdict(set)
    print(f"Analyzing {n} videos for duplicates (parallel comparisons)...")

    all_pairs = []
    added_pairs: Set[Tuple[str, str]] = set()

    all_buckets = sorted(duration_groups.keys())

    for bucket in all_buckets:
        bucket_features = duration_groups[bucket]
        bsize = len(bucket_features)

        if bsize >= 2:
            for i in range(bsize):
                for j in range(i + 1, bsize):
                    pair = tuple(sorted([bucket_features[i]["path"], bucket_features[j]["path"]]))
                    if pair not in added_pairs:
                        added_pairs.add(pair)
                        all_pairs.append((bucket_features[i], bucket_features[j]))

        adj_bucket = bucket + 30
        if adj_bucket in duration_groups:
            for f1 in bucket_features:
                for f2 in duration_groups[adj_bucket]:
                    pair = tuple(sorted([f1["path"], f2["path"]]))
                    if pair not in added_pairs:
                        added_pairs.add(pair)
                        all_pairs.append((f1, f2))

    if not all_pairs:
        return []

    n_workers = os.cpu_count() or 4
    chunk_size = min(5000, len(all_pairs))
    comparison_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(all_pairs), chunk_size):
            chunk = all_pairs[chunk_start:chunk_start + chunk_size]
            futures = {executor.submit(compare_features, f1, f2, verbose): (f1, f2) for f1, f2 in chunk}
            for future in tqdm(as_completed(futures), total=len(chunk),
                               desc=f"Comparing [{chunk_start}-{chunk_start + len(chunk)}]"):
                try:
                    f1, f2 = futures[future]
                    is_dup, _ = future.result()
                    if is_dup:
                        comparison_results.append((f1["path"], f2["path"]))
                except Exception:
                    pass

    for p1, p2 in comparison_results:
        matches[p1].add(p2)
        matches[p2].add(p1)

    visited: Set[str] = set()
    groups = []
    for f in features_list:
        vp = f["path"]
        if vp in visited:
            continue
        group = []
        queue = [vp]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            for neighbor in matches.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(group) > 1:
            groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Visual batch extraction
# ---------------------------------------------------------------------------

def get_video_resolution(video_path: str) -> Tuple[int, int]:
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        width, height = map(int, result.stdout.strip().split('x'))
        return (width, height)
    except Exception:
        return (0, 0)


def extract_visual_samples_batch(video_path: str, duration: float, temp_dir: str) -> List[Image.Image]:
    if temp_dir is None or duration <= 0:
        return []
    images = []
    base_name = os.path.basename(video_path)
    for i, point in enumerate([0.2, 0.4, 0.6, 0.8][:NUM_VISUAL_SAMPLES]):
        timestamp = duration * point
        try:
            temp_frame = os.path.join(temp_dir, f"frame_{base_name}_{i}.jpg")
            cmd = [
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-q:v', '2', temp_frame
            ]
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.exists(temp_frame):
                images.append(Image.open(temp_frame))
                os.remove(temp_frame)
        except Exception:
            pass
    return images


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

CACHE_DIR = ".dedup_cache"
CACHE_INDEX = os.path.join(CACHE_DIR, "index.json")
CACHE_LOCK = os.path.join(CACHE_DIR, ".lock")


def get_cache_key(video_path: str) -> str:
    if not os.path.exists(video_path):
        return ""
    stat = os.stat(video_path)
    key_str = f"{video_path}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def load_cache_index() -> Dict[str, Dict]:
    if os.path.exists(CACHE_INDEX):
        try:
            with open(CACHE_INDEX, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_cache_index(index: Dict[str, Dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    for attempt in range(5):
        try:
            with open(CACHE_LOCK, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    with open(CACHE_INDEX, 'w') as f:
                        json.dump(index, f, indent=2)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return
        except (IOError, OSError):
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))


def load_cached_features(video_path: str) -> Optional[VideoFeatures]:
    cache_key = get_cache_key(video_path)
    if not cache_key:
        return None
    index = load_cache_index()
    entry = index.get(cache_key)
    if entry is None:
        return None
    if os.path.abspath(entry.get("path", "")) != os.path.abspath(video_path):
        return None
    current_stat = os.stat(video_path)
    if entry.get("size") != current_stat.st_size or entry.get("mtime") != current_stat.st_mtime:
        return None
    features: VideoFeatures = {
        "path": video_path,
        "duration": entry.get("duration", 0.0),
        "resolution": tuple(entry.get("resolution", [0, 0])),
        "has_audio": entry.get("has_audio", False),
        "file_hash": entry.get("file_hash", ""),
        "audio_fingerprints": entry.get("audio_fingerprints", []),
        "visual_hashes": entry.get("visual_hashes", []),
        "file_size": entry.get("size", 0),
        "mtime": entry.get("mtime", 0.0),
    }
    return features


def save_features_to_cache(features: VideoFeatures) -> None:
    video_path = features.get("path", "")
    if not video_path:
        return
    cache_key = get_cache_key(video_path)
    if not cache_key:
        return
    index = load_cache_index()
    stat = os.stat(video_path)
    index[cache_key] = {
        "path": os.path.abspath(video_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "duration": features.get("duration", 0.0),
        "resolution": list(features.get("resolution", (0, 0))),
        "has_audio": features.get("has_audio", False),
        "file_hash": features.get("file_hash", ""),
        "visual_hashes": features.get("visual_hashes", []),
        "audio_fingerprints": features.get("audio_fingerprints", []),
    }
    save_cache_index(index)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_file_hash(video_path: str) -> str:
    """Compute xxHash (murmur64) of entire file for quick duplicate detection."""
    try:
        import xxhash
        h = xxhash.xxh64()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except ImportError:
        md5 = hashlib.md5()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                md5.update(chunk)
        return md5.hexdigest()


def extract_all_features(video_path: str, temp_dir: Optional[str] = None) -> VideoFeatures:
    """Extract all features from a video once, with cache support."""
    cached = load_cached_features(video_path)
    if cached and cached.get("duration", 0) > 0:
        return cached

    effective_temp_dir = temp_dir if temp_dir is not None else TEMP_DIR
    if effective_temp_dir is None:
        effective_temp_dir = tempfile.mkdtemp(prefix="video_dedup_")
    if not os.path.exists(effective_temp_dir):
        os.makedirs(effective_temp_dir, exist_ok=True)

    features: VideoFeatures = {
        "path": video_path,
        "duration": 0.0,
        "resolution": (0, 0),
        "has_audio": False,
        "audio_fingerprints": [],
        "visual_hashes": [],
        "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        "mtime": os.path.getmtime(video_path) if os.path.exists(video_path) else 0,
        "file_hash": compute_file_hash(video_path),
    }

    metadata = get_video_metadata(video_path)
    features["duration"] = metadata.get("duration", 0.0)
    features["resolution"] = metadata.get("resolution", (0, 0))
    features["has_audio"] = metadata.get("has_audio", False)

    if features["duration"] <= 0:
        return features

    duration = features["duration"]

    if features["has_audio"]:
        sample_starts: List[float] = []
        if duration > SKIP_FIRST_SECONDS + AUDIO_SAMPLE_DURATION:
            available = duration - SKIP_FIRST_SECONDS - AUDIO_SAMPLE_DURATION
            for i in range(NUM_AUDIO_SAMPLES):
                start = SKIP_FIRST_SECONDS + (available * i / max(NUM_AUDIO_SAMPLES - 1, 1))
                sample_starts.append(start)
        else:
            sample_starts = [max(0, duration / 2 - AUDIO_SAMPLE_DURATION / 2)]

        for start in sample_starts:
            audio_data = extract_audio_sample(video_path, start, AUDIO_SAMPLE_DURATION)
            if audio_data is not None:
                fp = generate_audio_fingerprint(audio_data)
                if len(fp) > 0:
                    features["audio_fingerprints"].append(fp.tolist())

    images = extract_visual_samples_batch(video_path, duration, effective_temp_dir)
    if images:
        features["visual_hashes"] = generate_visual_fingerprint(images)

    if features.get("duration", 0) > 0:
        save_features_to_cache(features)

    return features


# ---------------------------------------------------------------------------
# Upscaling detection
# ---------------------------------------------------------------------------

def calculate_frequency_score(image: np.ndarray) -> float:
    try:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        r2 = x * x + y * y
        low_energy = np.sum(magnitude[r2 <= (min(h, w) * 0.1) ** 2])
        mid_energy = np.sum(magnitude[(r2 > (min(h, w) * 0.1) ** 2) & (r2 <= (min(h, w) * 0.3) ** 2)])
        high_energy = np.sum(magnitude[r2 > (min(h, w) * 0.3) ** 2])
        total = low_energy + mid_energy + high_energy
        if total == 0:
            return 0.0
        high_ratio = high_energy / total
        expected = 0.15
        return 0.0 if high_ratio >= expected else min(1.0, (expected - high_ratio) / expected)
    except Exception:
        return 0.0


def calculate_edge_sharpness_score(image: np.ndarray) -> float:
    try:
        gray = (np.mean(image, axis=2) if len(image.shape) == 3 else image).astype(np.float32)
        laplacian_img = convolve(gray, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
        variance = np.var(laplacian_img) / (gray.shape[0] * gray.shape[1])
        h, w = gray.shape
        if max(h, w) >= 3840:
            expected = 1.5
        elif max(h, w) >= 1920:
            expected = 0.5
        else:
            return 0.0
        if variance >= expected * 0.6:
            return 0.0
        return max(0.0, min(1.0, 1.0 - variance / (expected * 0.6)))
    except Exception:
        return 0.0


def calculate_multiscale_score(video_path: str, width: int, height: int) -> float:
    try:
        if TEMP_DIR is None or max(width, height) < 1920:
            return 0.0
        duration = get_video_duration(video_path)
        timestamp = duration * 0.5
        temp_orig = os.path.join(TEMP_DIR, f"upscale_orig_{os.path.basename(video_path)}.jpg")
        temp_down = os.path.join(TEMP_DIR, f"upscale_down_{os.path.basename(video_path)}.jpg")

        subprocess.run(['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                        '-vframes', '1', '-q:v', '2', temp_orig],
                       capture_output=True, timeout=15)
        if not os.path.exists(temp_orig):
            return 0.0

        subprocess.run(['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                        '-vf', 'scale=1280:720:flags=lanczos', '-vframes', '1', '-q:v', '2', temp_down],
                       capture_output=True, timeout=15)
        if not os.path.exists(temp_down):
            os.remove(temp_orig)
            return 0.0

        arr_orig = np.array(Image.open(temp_orig).convert('RGB').resize(
            (1280, 720), Image.Resampling.LANCZOS)).astype(np.float32)
        arr_down = np.array(Image.open(temp_down).convert('RGB')).astype(np.float32)
        os.remove(temp_orig)
        os.remove(temp_down)

        std_o, std_d = np.std(arr_orig), np.std(arr_down)
        if std_o == 0 or std_d == 0:
            similarity = 1.0
        else:
            covariance = np.mean((arr_orig - arr_orig.mean()) * (arr_down - arr_down.mean()))
            similarity = max(0.0, min(1.0, (covariance / (std_o * std_d) + 1) / 2))

        return min(1.0, (similarity - 0.95) / 0.05) if similarity > 0.95 else 0.0
    except Exception:
        return 0.0


def detect_upscaling(video_path: str) -> Tuple[bool, float, Dict]:
    width, height = get_video_resolution(video_path)
    if max(width, height) < MIN_UPSCALING_ANALYSIS_RESOLUTION:
        return False, 0.0, {"skipped": True, "reason": "resolution_too_low",
                            "resolution": (int(width), int(height))}
    try:
        if TEMP_DIR is None:
            return False, 0.0, {"error": "temp_dir_not_set"}
        duration = get_video_duration(video_path)
        timestamp = max(SKIP_FIRST_SECONDS, duration * 0.3)
        temp_frame = os.path.join(TEMP_DIR, f"upscale_analysis_{os.path.basename(video_path)}.jpg")
        subprocess.run(['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                        '-vframes', '1', '-q:v', '2', temp_frame],
                       capture_output=True, timeout=15)
        if not os.path.exists(temp_frame):
            return False, 0.0, {"error": "frame_extraction_failed"}

        img_array = np.array(Image.open(temp_frame).convert('RGB'))
        os.remove(temp_frame)

        freq_score = calculate_frequency_score(img_array)
        edge_score = calculate_edge_sharpness_score(img_array)
        multi_score = calculate_multiscale_score(video_path, width, height)
        combined = 0.35 * freq_score + 0.25 * edge_score + 0.40 * multi_score
        is_upscaled = bool(combined > UPSCALING_CONFIDENCE_THRESHOLD)
        details = {
            "resolution": (int(width), int(height)),
            "scores": {
                "frequency": round(float(freq_score), 3),
                "edge_sharpness": round(float(edge_score), 3),
                "multiscale": round(float(multi_score), 3),
                "combined": round(float(combined), 3),
            },
            "threshold": float(UPSCALING_CONFIDENCE_THRESHOLD),
            "methods_used": ["frequency_analysis", "edge_sharpness", "multiscale_comparison"],
        }
        return is_upscaled, float(combined), details
    except Exception as e:
        return False, 0.0, {"error": str(e), "resolution": (int(width), int(height))}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_video_metadata(video_path: str) -> Dict:
    metadata: Dict[str, Any] = {
        "file_info": {}, "video": {}, "audio": [], "container": {}, "subtitles": {}
    }
    try:
        file_stat = os.stat(video_path)
        metadata["file_info"] = {
            "original_full_path": os.path.abspath(video_path),
            "filename": os.path.basename(video_path),
            "file_size_bytes": file_stat.st_size,
            "modification_time": file_stat.st_mtime,
        }
        cmd = ['ffprobe', '-v', 'error', '-show_streams', '-show_format',
               '-print_format', 'json', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return metadata
        probe_data = json.loads(result.stdout)

        video_stream = None
        audio_streams = []
        subtitle_streams = []
        for stream in probe_data.get('streams', []):
            ct = stream.get('codec_type', '')
            if ct == 'video':
                if not video_stream or stream.get('disposition', {}).get('default', 0):
                    video_stream = stream
            elif ct == 'audio':
                audio_streams.append(stream)
            elif ct == 'subtitle':
                subtitle_streams.append(stream)

        if video_stream:
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(video_stream.get('duration', 0) or
                             probe_data.get('format', {}).get('duration', 0) or 0)
            h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            except Exception:
                fps = 0
            pix_fmt = video_stream.get('pix_fmt', '')
            bit_depth = 10 if 'p10' in pix_fmt else 8
            color_transfer = video_stream.get('color_transfer', '').lower()
            hdr_format = None
            if any(f in color_transfer for f in ['smpte2084', 'arib-std-b67', 'smpte2086', 'cll']):
                hdr_format = ('HDR10' if '2084' in color_transfer
                              else 'HLG' if 'arib' in color_transfer else 'HDR')
            video_bitrate = 0
            if 'bit_rate' in video_stream:
                video_bitrate = int(video_stream['bit_rate']) // 1000
            metadata["video"] = {
                "width": width, "height": height, "resolution": f"{width}x{height}",
                "duration_seconds": round(duration, 2),
                "duration_formatted": f"{h:02d}:{m:02d}:{s:02d}",
                "video_codec": video_stream.get('codec_name', 'unknown'),
                "video_bitrate_kbps": video_bitrate,
                "frame_rate": round(fps, 2),
                "pixel_format": pix_fmt,
                "video_profile": video_stream.get('profile', 'unknown'),
                "bit_depth": bit_depth,
                "color_space": video_stream.get('color_space', '') or 'unknown',
                "hdr_format": hdr_format,
            }

        for ast in audio_streams:
            metadata["audio"].append({
                "codec": ast.get('codec_name', 'unknown'),
                "bitrate_kbps": int(ast.get('bit_rate', 0)) // 1000 if 'bit_rate' in ast else 0,
                "sample_rate_hz": int(ast.get('sample_rate', 0)),
                "channels": ast.get('channels', 0),
                "language": ast.get('tags', {}).get('language', 'und'),
            })

        fmt = probe_data.get('format', {})
        metadata["container"] = {
            "container_format": fmt.get('format_name', 'unknown').split(',')[0],
            "creation_time": fmt.get('tags', {}).get('creation_time'),
        }
        metadata["subtitles"] = {
            "subtitle_tracks_count": len(subtitle_streams),
            "subtitle_languages": [s.get('tags', {}).get('language', 'und') for s in subtitle_streams],
        }
    except Exception as e:
        print(f"Warning: Could not extract metadata for {video_path}: {e}")
    return metadata


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def calculate_quality_score(metadata: Dict) -> Tuple[int, str]:
    """
    Calculate quality score based on video characteristics.

    Priority order (highest to lowest weight):
      1. Resolution  — most impactful perceptual difference            (up to 65 pts)
      2. Frame rate  — smooth motion; weighted above bitrate because   (up to 25 pts)
                       bitrate can be artificially inflated by
                       re-encoding without any perceptible quality gain
      3. Codec       — efficiency tier affects real quality at same    (up to 20 pts)
                       bitrate
      4. HDR         — significant perceptual upgrade when present     (15 pts bonus)
      5. Bitrate     — useful signal but easily faked; capped low      (up to 15 pts)
                       to avoid preferring bloated re-encodes

    Returns (score, reason_string).
    """
    score = 0
    reasons = []

    video_info = metadata.get("video", {})
    if not video_info:
        return 0, "No video stream found"

    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    resolution_pixels = width * height

    # 1. Resolution — up to 65 pts
    score += int(40 * resolution_pixels / (1920 * 1080))
    if resolution_pixels >= 1920 * 1080:
        score += 25
        reasons.append(f"1080p+ ({width}x{height})")
    elif resolution_pixels >= 1280 * 720:
        score += 15
        reasons.append(f"720p ({width}x{height})")
    else:
        reasons.append(f"sub-720p ({width}x{height})")

    # 2. Frame rate — up to 25 pts
    fps = video_info.get("frame_rate", 0)
    if fps >= 59:
        score += 25; reasons.append("60fps")
    elif fps >= 49:
        score += 20; reasons.append("50fps")
    elif fps >= 29:
        score += 15; reasons.append("30fps")
    elif fps >= 24:
        score += 8;  reasons.append(f"{fps:.0f}fps")
    else:
        score += 3;  reasons.append(f"{fps:.0f}fps (low)")

    # 3. Codec — up to 20 pts
    codec = video_info.get("video_codec", "unknown").lower()
    if codec == 'av1':
        score += 20; reasons.append("AV1")
    elif codec in ('hevc', 'h265'):
        score += 18; reasons.append("HEVC/H.265")
    elif codec in ('h264', 'avc'):
        score += 13; reasons.append("H.264")
    elif codec in ('mpeg4', 'mpeg2', 'xvid', 'divx'):
        score += 6;  reasons.append(f"legacy ({codec})")
    else:
        score += 3;  reasons.append(f"codec:{codec}")

    # 4. HDR — 15 pt bonus
    hdr = video_info.get("hdr_format")
    if hdr:
        score += 15; reasons.append(f"HDR ({hdr})")

    # 5. Bitrate — up to 15 pts
    bitrate = video_info.get("video_bitrate_kbps", 0)
    if resolution_pixels > 0 and bitrate > 0:
        CODEC_EFF = {'av1': 3.0, 'hevc': 2.0, 'h265': 2.0, 'h264': 1.0, 'avc': 1.0}
        eff = CODEC_EFF.get(codec, 1.0)
        eff_br = int(bitrate * eff)
        optimal = (resolution_pixels / 1_000_000) * 2000
        ratio = eff_br / optimal
        if ratio < 0.3:
            score += int(15 * ratio / 0.3); q = "low bitrate"
        elif ratio < 1.0:
            score += int(5 + 7 * (ratio - 0.3) / 0.7); q = "adequate bitrate"
        else:
            score += min(15, int(12 * ratio ** 0.4)); q = "good bitrate"
        if eff > 1.0:
            reasons.append(f"{q} ({bitrate}kbps ×{eff:.0f} = {eff_br}kbps eff, {ratio:.2f}x opt)")
        else:
            reasons.append(f"{q} ({eff_br}kbps, {ratio:.2f}x opt)")
    else:
        reasons.append("unknown bitrate")

    return max(0, score), ", ".join(reasons)


def analyze_duplicate_set(videos_with_metadata: List[Tuple[str, Dict]]) -> Dict[str, Dict]:
    if not videos_with_metadata:
        return {}

    results = {}
    for video_path, metadata in videos_with_metadata:
        score, reason = calculate_quality_score(metadata)
        results[video_path] = {
            "metadata": metadata, "quality_score": score, "quality_reason": reason,
            "recommendation": "DELETE_CANDIDATE", "reason": "", "better_alternative": None,
        }

    if len(results) == 1:
        vp = list(results.keys())[0]
        results[vp]["recommendation"] = "KEEP"
        results[vp]["reason"] = "Only copy of this content"
        return results

    best_path = max(
        results.keys(),
        key=lambda p: (
            results[p]["quality_score"],
            results[p]["metadata"].get("video", {}).get("width", 0),
            results[p]["metadata"].get("video", {}).get("frame_rate", 0),
            results[p]["metadata"].get("video", {}).get("video_bitrate_kbps", 0),
        )
    )
    best_score = results[best_path]["quality_score"]
    bvi = results[best_path]["metadata"].get("video", {})
    best_res = bvi.get("resolution", "unknown")
    best_codec = bvi.get("video_codec", "unknown")
    best_fps = bvi.get("frame_rate", 0)
    best_bitrate = bvi.get("video_bitrate_kbps", 0)
    best_width = bvi.get("width", 0)
    best_height = bvi.get("height", 0)

    results[best_path]["recommendation"] = "KEEP"
    results[best_path]["reason"] = (
        f"Best quality: {best_res}, {best_codec}, {best_fps}fps, score {best_score}"
    )

    for video_path, result in results.items():
        if video_path == best_path:
            continue
        score = result["quality_score"]
        vi = result["metadata"].get("video", {})
        v_fps = vi.get("frame_rate", 0)
        v_codec = vi.get("video_codec", "unknown")
        v_brate = vi.get("video_bitrate_kbps", 0)
        v_width = vi.get("width", 0)
        v_height = vi.get("height", 0)

        reason = (f"Significantly lower quality (score {score} vs {best_score})"
                  if best_score - score >= 30 else
                  f"Lower quality (score {score} vs {best_score})")
        if v_width * v_height < best_width * best_height:
            reason += f"; lower resolution ({v_width}x{v_height} vs {best_width}x{best_height})"
        if v_fps < best_fps:
            reason += f"; lower frame rate ({v_fps}fps vs {best_fps}fps)"
        if v_codec != best_codec and best_codec in ('av1', 'hevc', 'h265'):
            reason += f"; less efficient codec ({v_codec} vs {best_codec})"
        if v_brate < best_bitrate * 0.7 and v_brate > 0:
            reason += f"; lower bitrate ({v_brate} vs {best_bitrate} kbps)"

        result["recommendation"] = "DELETE_CANDIDATE"
        result["reason"] = reason
        result["better_alternative"] = {
            "filename": os.path.basename(best_path),
            "quality_score": best_score,
            "reason": f"Better quality: {best_res}, {best_codec}, {best_fps}fps, score {best_score}",
        }

    return results


# ---------------------------------------------------------------------------
# Organise duplicates
# ---------------------------------------------------------------------------

def organize_duplicates(
    directory: str,
    duplicate_groups: List[List[str]],
    dry_run: bool = False,
    create_markers: bool = False,
) -> None:
    if not duplicate_groups:
        print("No duplicates found!")
        return

    print(f"\nFound {len(duplicate_groups)} duplicate sets")
    deduped_base = os.path.join(directory, "__deduped")

    for i, group in enumerate(duplicate_groups, 1):
        folder_name = f"duplicate_set_{i:03d}"
        folder_path = os.path.join(deduped_base, folder_name)
        print(f"\n{folder_name}: {len(group)} videos")

        videos_with_metadata = []
        for video_path in group:
            print(f"  Extracting metadata: {os.path.basename(video_path)}")
            videos_with_metadata.append((video_path, extract_video_metadata(video_path)))

        analysis_results = analyze_duplicate_set(videos_with_metadata)

        for video_path in group:
            rel_path = os.path.relpath(video_path, directory)
            analysis = analysis_results.get(video_path, {})
            print(f"  - {rel_path} [{analysis.get('recommendation', 'UNKNOWN')}, "
                  f"score: {analysis.get('quality_score', 0)}]")

        if not dry_run:
            os.makedirs(folder_path, exist_ok=True)
            for video_path in group:
                try:
                    filename = os.path.basename(video_path)
                    dest_path = os.path.join(folder_path, filename)
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(folder_path, f"{base_name}_{counter:02d}{ext}")
                        counter += 1

                    analysis = analysis_results.get(video_path, {})
                    metadata = analysis.get("metadata", {})
                    json_data = {
                        "recommendation": analysis.get("recommendation", "UNKNOWN"),
                        "original_full_path": (metadata.get("file_info", {})
                                               .get("original_full_path", os.path.abspath(video_path))),
                        "quality_score": analysis.get("quality_score", 0),
                        "reason": analysis.get("reason", ""),
                        "filename": filename,
                        "file_size_bytes": metadata.get("file_info", {}).get("file_size_bytes", 0),
                        "modification_time": metadata.get("file_info", {}).get("modification_time", 0),
                        **metadata,
                    }
                    if analysis.get("recommendation") == "DELETE_CANDIDATE":
                        json_data["better_alternative"] = analysis.get("better_alternative")

                    json_filename = f"{os.path.basename(dest_path)}.json"
                    with open(os.path.join(folder_path, json_filename), 'w') as f:
                        json.dump(json_data, f, indent=2)

                    if create_markers:
                        rec = analysis.get("recommendation", "")
                        ext_marker = ".keep" if rec == "KEEP" else ".delete" if rec == "DELETE_CANDIDATE" else None
                        if ext_marker:
                            open(os.path.join(folder_path,
                                              f"{os.path.basename(dest_path)}{ext_marker}"), 'w').close()

                    shutil.move(video_path, dest_path)
                    print(f"  -> Moved to __deduped/{folder_name}")
                    print(f"  -> Created metadata: {json_filename}")
                except Exception as e:
                    print(f"  ERROR moving {video_path}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global TEMP_DIR

    parser = argparse.ArgumentParser(description='Find and organize duplicate videos')
    parser.add_argument('directory', help='Directory containing videos')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show duplicates without moving files')
    parser.add_argument('--report', type=str, default='duplicates_report.json',
                        help='Path to save JSON report')
    parser.add_argument('--detect-upscaling', action='store_true',
                        help='Detect videos upscaled from lower resolution')
    parser.add_argument('--create-markers', action='store_true',
                        help='Create .keep/.delete marker files in __deduped folders')
    parser.add_argument('--include-subfolders', nargs='*', metavar='PATH',
                        help='Include subfolders. No args = all; with args = only those paths.')
    parser.add_argument('--exclude-root', action='store_true',
                        help='Exclude root directory (requires --include-subfolders)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show verbose output for comparison stages (debugging)')

    args = parser.parse_args()
    directory = os.path.abspath(args.directory)

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    if args.exclude_root and args.include_subfolders is None:
        print("Error: --exclude-root requires --include-subfolders to be specified")
        sys.exit(1)

    TEMP_DIR = tempfile.mkdtemp(prefix="video_dedup_")
    upscaling_results: Dict[str, Any] = {}

    try:
        include_subfolders = args.include_subfolders
        exclude_root = args.exclude_root

        if include_subfolders is not None:
            if len(include_subfolders) == 0:
                print(f"Scanning {directory} and all subfolders...")
            else:
                print(f"Scanning {directory} with specific subfolders:")
                for path in include_subfolders:
                    print(f"  - {path}")
        else:
            print(f"Scanning {directory} (root directory only)...")

        if exclude_root:
            print("  (excluding root directory)")

        video_paths = find_videos(directory, include_subfolders=include_subfolders,
                                  exclude_root=exclude_root)
        if not video_paths:
            print("No videos found!")
            sys.exit(0)

        video_paths = sorted(video_paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
        print(f"Found {len(video_paths)} videos")

        if args.detect_upscaling:
            print("\n" + "=" * 60)
            print("UPSCALING DETECTION MODE")
            print("=" * 60)
            for path in tqdm(video_paths, desc="Analyzing upscaling"):
                is_upscaled, confidence, details = detect_upscaling(path)
                upscaling_results[path] = {
                    "is_upscaled": is_upscaled,
                    "confidence": round(confidence, 3),
                    "details": details,
                }
            upscaled_count = sum(1 for r in upscaling_results.values() if r["is_upscaled"])
            print(f"\nUpscaling: {upscaled_count} of {len(upscaling_results)} flagged")
            for path, result in upscaling_results.items():
                if result["is_upscaled"]:
                    w, h = result["details"].get("resolution", (0, 0))
                    print(f"  - {os.path.basename(path)} ({w}x{h}) [conf: {result['confidence']:.2f}]")

        n_workers = os.cpu_count() or 4
        print(f"Using {n_workers} workers for feature extraction")

        features_list: List[VideoFeatures] = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_all_features, path): path for path in video_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                try:
                    features = future.result()
                    if features.get("duration", 0) > 0:
                        features_list.append(features)
                except Exception:
                    pass

        if len(features_list) < 2:
            print("Not enough valid videos to compare")
            sys.exit(0)

        duplicate_groups = find_duplicate_groups_with_features(features_list, verbose=args.verbose)

        report: Dict[str, Any] = {
            'directory': directory,
            'total_videos': len(features_list),
            'duplicate_sets': len(duplicate_groups),
            'sets': [
                {
                    'set_id': i + 1,
                    'videos': [
                        {
                            'path': v,
                            'filename': os.path.basename(v),
                            'duration': next((f.get("duration", 0)
                                             for f in features_list if f["path"] == v), 0),
                        }
                        for v in group
                    ],
                }
                for i, group in enumerate(duplicate_groups)
            ],
        }

        if args.detect_upscaling and upscaling_results:
            report['upscaling_analysis'] = {
                'enabled': True,
                'threshold': UPSCALING_CONFIDENCE_THRESHOLD,
                'min_resolution_analyzed': MIN_UPSCALING_ANALYSIS_RESOLUTION,
                'summary': {
                    'total_analyzed': len(upscaling_results),
                    'upscaled_detected': sum(1 for r in upscaling_results.values() if r['is_upscaled']),
                    'analysis_methods': ['frequency_analysis', 'edge_sharpness', 'multiscale_comparison'],
                },
                'videos': [
                    {
                        'path': path, 'filename': os.path.basename(path),
                        'is_upscaled': result['is_upscaled'],
                        'confidence': result['confidence'],
                        'resolution': result['details'].get('resolution', [0, 0]),
                        'scores': result['details'].get('scores', {}),
                    }
                    for path, result in upscaling_results.items()
                ],
            }

        report_path = os.path.join(directory, args.report)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        organize_duplicates(directory, duplicate_groups, args.dry_run, args.create_markers)

    finally:
        if TEMP_DIR and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


if __name__ == '__main__':
    main()