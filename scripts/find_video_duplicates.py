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


def compare_features(f1: VideoFeatures, f2: VideoFeatures, verbose: bool = False, visual_threshold: float = None) -> Tuple[bool, str]:
    """Compare two videos using precomputed features with multi-stage filtering."""
    # Use provided threshold or fall back to global default
    threshold = visual_threshold if visual_threshold is not None else VISUAL_SIM_THRESHOLD
    
    dur1, dur2 = f1.get("duration", 0), f2.get("duration", 0)
    v1_name = os.path.basename(f1.get("path", ""))
    v2_name = os.path.basename(f2.get("path", ""))
    
    if dur1 <= 0 or dur2 <= 0:
        return False, "no_duration"
    
    # STAGE 0: Quick file hash (instant - identical files)
    hash1 = f1.get("file_hash", "")
    hash2 = f2.get("file_hash", "")
    if hash1 and hash2 and hash1 == hash2:
        if verbose:
            print(f"  {v1_name} vs {v2_name}")
            print(f"    Stage 0 (File Hash): MATCH (identical files)")
        return True, "file_hash"
    
    # STAGE 1: Duration (fastest)
    duration_diff = abs(dur1 - dur2) / max(dur1, dur2, 1)
    if verbose:
        print(f"  {v1_name} vs {v2_name}")
        print(f"    Stage 1 (Duration): diff={duration_diff:.4f}, threshold={LENGTH_TOLERANCE}, result={'PASS' if duration_diff <= LENGTH_TOLERANCE else 'FAIL'}")
    if duration_diff > LENGTH_TOLERANCE:
        return False, "duration_mismatch"
    
    # STAGE 2: Resolution check (aspect ratio tolerance for different encodings)
    res1 = f1.get("resolution", (0, 0))
    res2 = f2.get("resolution", (0, 0))
    stage2_pass = True
    if res1 != (0, 0) and res2 != (0, 0):
        ratio1 = res1[0] / res1[1] if res1[1] > 0 else 0
        ratio2 = res2[0] / res2[1] if res2[1] > 0 else 0
        ratio_diff = abs(ratio1 - ratio2)
        stage2_pass = ratio_diff <= 0.15
        if verbose:
            print(f"    Stage 2 (Resolution): ratio1={ratio1:.3f}, ratio2={ratio2:.3f}, diff={ratio_diff:.4f}, threshold=0.15, result={'PASS' if stage2_pass else 'FAIL'}")
        if not stage2_pass:
            return False, "resolution_mismatch"
    
    # STAGE 3: File size check (more permissive - same content can have vastly different sizes)
    size1 = f1.get("file_size", 0)
    size2 = f2.get("file_size", 0)
    stage3_pass = True
    if size1 > 0 and size2 > 0:
        size_diff = abs(size1 - size2) / max(size1, size2)
        stage3_pass = size_diff <= 0.9
        if verbose:
            print(f"    Stage 3 (Size): size1={size1}, size2={size2}, diff={size_diff:.4f}, threshold=0.9, result={'PASS' if stage3_pass else 'FAIL'}")
        if not stage3_pass:
            return False, "size_mismatch"
    
    # STAGE 4: Audio fingerprint (expensive)
    has_audio1 = f1.get("has_audio", False)
    has_audio2 = f2.get("has_audio", False)
    audio_pass = False
    audio_details = ""
    if has_audio1 and has_audio2:
        fps1 = f1.get("audio_fingerprints", [])
        fps2 = f2.get("audio_fingerprints", [])
        
        if fps1 and fps2:
            matches = 0
            match_details = []
            for fp1_list in fps1:
                fp1_arr = np.array(fp1_list)
                best_sim = 0.0
                for fp2_list in fps2:
                    fp2_arr = np.array(fp2_list)
                    sim = compare_audio_fingerprints(fp1_arr, fp2_arr)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim > 0.95:
                    matches += 1
                match_details.append(f"{best_sim:.2f}")
            
            audio_pass = matches >= len(fps1)
            audio_details = f"matches={matches}/{len(fps1)} (thresholds: {', '.join(match_details)})"
            if verbose:
                print(f"    Stage 4 (Audio): {audio_details}, result={'PASS' if audio_pass else 'FAIL'}")
            
            if audio_pass:
                return True, "audio_fingerprint"
        else:
            if verbose:
                has_audio1 = f1.get("has_audio", False)
                has_audio2 = f2.get("has_audio", False)
                fps1 = f1.get("audio_fingerprints", [])
                fps2 = f2.get("audio_fingerprints", [])
                has_fps1 = "YES" if fps1 else "NO"
                has_fps2 = "YES" if fps2 else "NO"
                print(f"    Stage 4 (Audio): has_audio={has_audio1}/{has_audio2}, fps={has_fps1}/{has_fps2}, result=SKIP")
    
    # STAGE 5: Visual hash (expensive)
    hashes1 = f1.get("visual_hashes", [])
    hashes2 = f2.get("visual_hashes", [])
    visual_pass = False
    if verbose:
        # Debug: show visual hash details
        h1_count = len(hashes1) if hashes1 else 0
        h2_count = len(hashes2) if hashes2 else 0
        h1_regions = len(hashes1[0]) if hashes1 and hashes1[0] else 0
        h2_regions = len(hashes2[0]) if hashes2 and hashes2[0] else 0
        print(f"    Stage 5 (Visual): frames1={h1_count}, frames2={h2_count}, regions_per_frame={h1_regions}")
    
    if hashes1 and hashes2:
        try:
            visual_sim = compare_visual_fingerprints(hashes1, hashes2, verbose)
        except Exception as e:
            if verbose:
                print(f"    Stage 5 (Visual): ERROR in comparison: {e}")
                print(f"      hashes1: {len(hashes1)} frames, first={hashes1[0] if hashes1 else 'empty'}")
                print(f"      hashes2: {len(hashes2)} frames, first={hashes2[0] if hashes2 else 'empty'}")
            visual_sim = 0.0
        visual_pass = visual_sim >= threshold
        if verbose:
            print(f"    Stage 5 (Visual): similarity={visual_sim:.4f}, threshold={threshold}, result={'PASS' if visual_pass else 'FAIL'}")
        if visual_pass:
            return True, "visual_fingerprint"
    else:
        if verbose:
            has_h1 = "YES" if hashes1 else "NO"
            has_h2 = "YES" if hashes2 else "NO"
            print(f"    Stage 5 (Visual): hashes1={has_h1}, hashes2={has_h2}, result=SKIP (no hashes)")
    
    return False, "no_match"

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}

# Configuration
LENGTH_TOLERANCE = 0.15  # 15% length difference allowed
AUDIO_SAMPLE_DURATION = 5  # seconds per audio sample
NUM_AUDIO_SAMPLES = 4  # number of samples to extract
NUM_VISUAL_SAMPLES = 4  # number of frames to extract (default)
VISUAL_SIM_THRESHOLD = 0.7  # minimum visual similarity threshold
SKIP_FIRST_SECONDS = 10  # skip first 10 seconds to avoid intros/ads
TEMP_DIR = None

# Upscaling Detection Configuration
UPSCALING_CONFIDENCE_THRESHOLD = 0.65  # Conservative threshold to avoid false positives on soft videos
MIN_UPSCALING_ANALYSIS_RESOLUTION = 720  # Only analyze videos >= 720p


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Get all video metadata in a single ffprobe call."""
    result = {
        "duration": 0.0,
        "resolution": (0, 0),
        "has_audio": False,
    }
    
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=width,height,codec_type',
            '-of', 'json',
            video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)
        
        # Parse duration
        if "format" in data and "duration" in data["format"]:
            result["duration"] = float(data["format"]["duration"])
        
        # Parse streams
        if "streams" in data:
            for stream in data["streams"]:
                if stream.get("codec_type") == "video":
                    w = stream.get("width", 0)
                    h = stream.get("height", 0)
                    result["resolution"] = (w, h)
                elif stream.get("codec_type") == "audio":
                    result["has_audio"] = True
        
    except Exception:
        pass
    
    return result


def extract_audio_sample(video_path: str, start_time: float, duration: float) -> Optional[np.ndarray]:
    """Extract audio sample from video at specific time."""
    if TEMP_DIR is None:
        return None
    
    try:
        # Create temp wav file
        temp_wav = os.path.join(TEMP_DIR, f"audio_{os.path.basename(video_path)}_{start_time}.wav")
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
            '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            temp_wav
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        
        if not os.path.exists(temp_wav):
            return None
        
        # Read audio data
        with audioread.audio_open(temp_wav) as f:
            audio_data = []
            for buf in f:
                # Convert bytes to numpy array
                data = np.frombuffer(buf, dtype=np.int16)
                audio_data.append(data)
            
            if audio_data:
                audio_array = np.concatenate(audio_data)
                # Clean up temp file
                os.remove(temp_wav)
                return audio_array
        
        return None
    except Exception as e:
        return None


def generate_audio_fingerprint(audio_data: np.ndarray) -> np.ndarray:
    """Generate a simple audio fingerprint using spectral features."""
    # Normalize
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Compute spectrogram
    n_fft = 512
    hop_length = 256
    
    # Pad if necessary
    if len(audio_data) < n_fft:
        audio_data = np.pad(audio_data, (0, n_fft - len(audio_data)))
    
    # Compute STFT
    frames = []
    for i in range(0, len(audio_data) - n_fft + 1, hop_length):
        frame = audio_data[i:i + n_fft]
        windowed = frame * np.hanning(n_fft)
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        frames.append(magnitude)
    
    if not frames:
        return np.array([])
    
    spectrogram = np.array(frames).T
    
    # Extract peaks (simplified chromaprint-like approach)
    # Divide into frequency bands and find peaks in each time window
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
                    peak_idx = np.argmax(band_energy)
                    fingerprint.append(band_energy[peak_idx])
    
    # Normalize
    if fingerprint:
        fingerprint = np.array(fingerprint)
        fingerprint = fingerprint / (np.max(fingerprint) + 1e-10)
    
    return np.array(fingerprint)


def has_audio_stream(video_path: str) -> bool:
    """Check if video has an audio stream."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return 'audio' in result.stdout.lower()
    except:
        return False


def generate_visual_fingerprint(images: List[Image.Image]) -> List[List[str]]:
    """Generate region-based perceptual hashes per frame (top/middle/bottom)."""
    hashes = []
    for img in images:
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            regions = [
                img.crop((0, 0, w, h // 3)),           # top
                img.crop((0, h // 3, w, 2 * h // 3)),  # middle
                img.crop((0, 2 * h // 3, w, h))        # bottom
            ]
            region_hashes = [str(imagehash.phash(r)) for r in regions]
            hashes.append(region_hashes)
        except Exception:
            pass
    return hashes


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hex hashes."""
    if len(hash1) != len(hash2):
        return 1000  # Large distance if different lengths
    
    # Convert hex to binary
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    
    # Count differences
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def compare_audio_fingerprints(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compare two audio fingerprints and return similarity (0-1)."""
    if len(fp1) == 0 or len(fp2) == 0:
        return 0.0
    
    # Normalize lengths
    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]
    
    # Compute correlation
    std1 = np.std(fp1)
    std2 = np.std(fp2)
    if std1 == 0 or std2 == 0:
        return 0.0
    
    # Reject low-variance fingerprints (indicates silence/static)
    MIN_FINGERPRINT_VARIANCE = 0.1
    if std1 < MIN_FINGERPRINT_VARIANCE or std2 < MIN_FINGERPRINT_VARIANCE:
        return 0.0
    
    correlation = np.corrcoef(fp1, fp2)[0, 1]
    if np.isnan(correlation):
        return 0.0
    
    # Convert to 0-1 range
    similarity = (correlation + 1) / 2
    return max(0.0, similarity)


def compare_visual_fingerprints(hashes1: List[List[str]], hashes2: List[List[str]], verbose: bool = False) -> float:
    """Improved version - tolerant to small timing shifts and re-encoding."""
    if verbose:
        print(f"    DEBUG: compare_visual_fingerprints called with {len(hashes1)} vs {len(hashes2)} frame hashes")

    if not hashes1 or not hashes2:
        return 0.0

    n = min(len(hashes1), len(hashes2))
    if n == 0:
        return 0.0

    threshold_match = 9
    threshold_bad = 26
    good_frames = 0

    for i in range(n):
        regions1 = hashes1[i]
        regions2 = hashes2[i]
        region_matches = 0
        dists = [hamming_distance(r1, r2) for r1, r2 in zip(regions1, regions2)]
        max_dist = max(dists)

        for dist in dists:
            if dist <= threshold_match:
                region_matches += 1

        is_good = (region_matches >= 2) and (max_dist <= threshold_bad)

        if verbose:
            status = "GOOD" if is_good else "BAD"
            print(f"      Frame {i}: top:{dists[0]} mid:{dists[1]} bot:{dists[2]} → {status} ({region_matches}/3)")

        if is_good:
            good_frames += 1

    similarity = good_frames / n

    if verbose:
        print(f"    Final visual similarity: {similarity:.4f} ({good_frames}/{n} good frames)")

    return similarity


def find_videos(
    directory: str,
    include_subfolders: Optional[List[str]] = None,
    exclude_root: bool = False,
    include_hidden: bool = False
) -> List[str]:
    """
    Find all video files in directory.
    
    Args:
        directory: Base directory to scan
        include_subfolders: None = no subfolders, empty list = all subfolders (except __deduped),
                           list of paths = only those specific subfolders
        exclude_root: If True, don't include videos from the root directory
        include_hidden: If True, include hidden subfolders (starting with .)
    
    Returns:
        List of video file paths
    """
    videos = []
    abs_directory = os.path.abspath(directory)
    
    # Always exclude the __deduped folder
    deduped_folder = os.path.join(abs_directory, "__deduped")
    
    def is_in_deduped(path: str) -> bool:
        """Check if path is inside the __deduped folder."""
        try:
            return os.path.commonpath([path, deduped_folder]) == deduped_folder or \
                   path.startswith(deduped_folder + os.sep)
        except ValueError:
            return False
    
    if include_subfolders is None:
        # Only scan root directory
        if not exclude_root:
            for file in os.listdir(abs_directory):
                file_path = os.path.join(abs_directory, file)
                if os.path.isfile(file_path):
                    ext = Path(file).suffix.lower()
                    if ext in VIDEO_EXTENSIONS:
                        videos.append(file_path)
    else:
        # include_subfolders is a list (possibly empty)
        if len(include_subfolders) == 0:
            # Include all subfolders recursively, but exclude __deduped
            for root, dirs, files in os.walk(abs_directory):
                # Skip if this directory is inside __deduped
                if is_in_deduped(root):
                    continue
                
                # Filter out hidden directories unless include_hidden is True
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                # Skip root directory if exclude_root is True
                if exclude_root and root == abs_directory:
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = Path(file).suffix.lower()
                    if ext in VIDEO_EXTENSIONS:
                        videos.append(file_path)
        else:
            # Include only specified subfolders + root (if not excluded)
            folders_to_scan = []
            
            # Add root if not excluded
            if not exclude_root:
                folders_to_scan.append(abs_directory)
            
            # Add specified subfolders
            for subfolder in include_subfolders:
                # Resolve path - absolute paths are used as-is, relative paths resolved against base
                if os.path.isabs(subfolder):
                    subfolder_path = os.path.abspath(subfolder)
                else:
                    subfolder_path = os.path.abspath(os.path.join(abs_directory, subfolder))
                
                # Validate it exists and is a directory
                if not os.path.exists(subfolder_path):
                    print(f"Warning: Specified subfolder does not exist: {subfolder}")
                    continue
                if not os.path.isdir(subfolder_path):
                    print(f"Warning: Specified path is not a directory: {subfolder}")
                    continue
                
                # Skip if it's the __deduped folder
                if is_in_deduped(subfolder_path):
                    print(f"Warning: Skipping __deduped folder: {subfolder}")
                    continue
                
                folders_to_scan.append(subfolder_path)
            
            # Scan each folder recursively
            for folder in folders_to_scan:
                for root, dirs, files in os.walk(folder):
                    # Skip if this directory is inside __deduped
                    if is_in_deduped(root):
                        continue
                    
                    # Filter out hidden directories unless include_hidden is True
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        ext = Path(file).suffix.lower()
                        if ext in VIDEO_EXTENSIONS:
                            videos.append(file_path)
    
    return videos


def find_duplicate_groups_with_features(features_list: List[VideoFeatures], verbose: bool = False) -> List[List[str]]:
    """Find groups of duplicate videos using precomputed features with parallel comparisons."""
    n = len(features_list)
    if n == 0:
        return []
    
    duration_groups: Dict[int, List[VideoFeatures]] = defaultdict(list)
    for f in features_list:
        bucket = round(f.get("duration", 0) / 30) * 30
        duration_groups[bucket].append(f)
    
    # Debug: Show bucket distribution
    if verbose:
        print(f"\n[DEBUG] Bucket distribution ({len(duration_groups)} buckets):")
        for bucket in sorted(duration_groups.keys()):
            videos = [os.path.basename(f["path"]) for f in duration_groups[bucket]]
            print(f"  Bucket {bucket}s: {len(videos)} videos - {videos[:3]}{'...' if len(videos) > 3 else ''}")
    
    matches = defaultdict(set)
    
    print(f"Analyzing {n} videos for duplicates (parallel comparisons)...")
    
    # Generate all comparison pairs (within bucket + adjacent buckets for edge cases)
    all_pairs = []
    all_buckets = sorted(duration_groups.keys())
    added_pairs = set()  # Track added pairs to avoid duplicates
    
    for i, bucket in enumerate(all_buckets):
        bucket_features = duration_groups[bucket]
        bucket_size = len(bucket_features)
        
        # Within bucket pairs (only if 2+ videos)
        if bucket_size >= 2:
            for j in range(bucket_size):
                for k in range(j + 1, bucket_size):
                    all_pairs.append((bucket_features[j], bucket_features[k]))
        
        # Compare with adjacent bucket (+30s only) - ALWAYS, regardless of bucket_size
        adj_bucket = bucket + 30
        if adj_bucket in duration_groups:
            adj_features = duration_groups[adj_bucket]
            for f1 in bucket_features:
                for f2 in adj_features:
                    # Create a unique key for this pair (sorted paths)
                    pair_key = tuple(sorted([f1["path"], f2["path"]]))
                    if pair_key not in added_pairs:
                        added_pairs.add(pair_key)
                        all_pairs.append((f1, f2))
    
    # Debug: Show comparison pairs for specific videos
    if verbose:
        target_names = ["Hello"]
        print(f"\n[DEBUG] Generated {len(all_pairs)} comparison pairs")
        print(f"[DEBUG] Looking for pairs involving: {target_names}")
        for f1, f2 in all_pairs:
            name1 = os.path.basename(f1["path"])
            name2 = os.path.basename(f2["path"])
            if any(t.lower() in name1.lower() for t in target_names) or any(t.lower() in name2.lower() for t in target_names):
                print(f"  PAIR: {name1[:50]} vs {name2[:50]}")
    
    if not all_pairs:
        return []
    
    # Parallel comparison with chunking for memory efficiency
    n_workers = os.cpu_count() or 4
    comparison_results = []
    chunk_size = min(5000, len(all_pairs))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(all_pairs), chunk_size):
            chunk = all_pairs[chunk_start:chunk_start + chunk_size]
            futures = {executor.submit(compare_features, f1, f2, verbose, VISUAL_SIM_THRESHOLD): (f1, f2) for f1, f2 in chunk}
            for future in tqdm(as_completed(futures), total=len(chunk), desc=f"Comparing [{chunk_start}-{chunk_start+len(chunk)}]"):
                try:
                    f1, f2 = futures[future]
                    is_dup, method = future.result()
                    if is_dup:
                        comparison_results.append((f1["path"], f2["path"]))
                except Exception:
                    pass
    
    # Build match graph
    for p1, p2 in comparison_results:
        matches[p1].add(p2)
        matches[p2].add(p1)
    
    # Find connected components
    visited = set()
    groups = []
    for f in features_list:
        video_path = f["path"]
        if video_path in visited:
            continue
        group = []
        queue = [video_path]
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


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Get video width and height using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        width, height = map(int, result.stdout.strip().split('x'))
        return (width, height)
    except Exception as e:
        return (0, 0)


def extract_visual_samples_batch(video_path: str, duration: float, temp_dir: str) -> List[Image.Image]:
    """Extract multiple frames - improved sampling for small timing differences."""
    if temp_dir is None or duration <= 0:
        return []
    
    # Calculate sample points based on NUM_VISUAL_SAMPLES
    num_samples = NUM_VISUAL_SAMPLES
    if num_samples == 4:
        sample_points = [0.20, 0.35, 0.55, 0.75]
    elif num_samples == 7:
        sample_points = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]
    else:
        step = 0.70 / (num_samples - 1) if num_samples > 1 else 0
        sample_points = [0.15 + i * step for i in range(num_samples)]
    
    images = []
    base_name = os.path.basename(video_path)
    
    for i, point in enumerate(sample_points):
        timestamp = duration * point
        try:
            temp_frame = os.path.join(temp_dir, f"frame_{base_name}_{i:02d}.jpg")
            
            cmd = [
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-q:v', '2', temp_frame
            ]
            subprocess.run(cmd, capture_output=True, timeout=15)
            
            if os.path.exists(temp_frame):
                img = Image.open(temp_frame)
                images.append(img)
                os.remove(temp_frame)
        except Exception:
            pass
    
    return images


CACHE_DIR = ".dedup_cache"
CACHE_INDEX = os.path.join(CACHE_DIR, "index.json")
CACHE_LOCK = os.path.join(CACHE_DIR, ".lock")


def get_cache_key(video_path: str) -> str:
    """Generate cache key from path + size + mtime."""
    if not os.path.exists(video_path):
        return ""
    stat = os.stat(video_path)
    key_str = f"{video_path}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


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


# Check and log which hash method is available
try:
    import xxhash
    HASH_METHOD = "xxhash"
except ImportError:
    HASH_METHOD = "MD5"


def load_cache_index() -> Dict[str, Dict]:
    """Load cache index from disk."""
    if os.path.exists(CACHE_INDEX):
        try:
            with open(CACHE_INDEX, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_cache_index(index: Dict[str, Dict]) -> None:
    """Save cache index to disk with locking."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    max_retries = 5
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                pass


def load_cached_features(video_path: str) -> Optional[VideoFeatures]:
    """Load features from cache if valid, else None."""
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
        "audio_fingerprints": [],
        "visual_hashes": entry.get("visual_hashes", []),
        "file_size": entry.get("size", 0),
        "mtime": entry.get("mtime", 0.0),
    }
    
    audio_fp_list = entry.get("audio_fingerprints", [])
    if audio_fp_list:
        features["audio_fingerprints"] = audio_fp_list
    
    return features


def save_features_to_cache(features: VideoFeatures) -> None:
    """Save features to cache."""
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


def extract_all_features(video_path: str, temp_dir: Optional[str] = None) -> VideoFeatures:
    """
    Extract all features from a video ONCE.
    Returns a dict with: duration, resolution, has_audio, audio_fingerprints, visual_hashes.
    
    Args:
        video_path: Path to the video file
        temp_dir: Optional temp directory. If None, uses global TEMP_DIR (for backward compatibility)
    """
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
    }
    
    if effective_temp_dir is None:
        return features
    
    # Single ffprobe call for all metadata
    metadata = get_video_metadata(video_path)
    features["duration"] = metadata.get("duration", 0.0)
    features["resolution"] = metadata.get("resolution", (0, 0))
    features["has_audio"] = metadata.get("has_audio", False)
    
    # Compute file hash for quick duplicate detection
    features["file_hash"] = compute_file_hash(video_path)
    
    if features["duration"] <= 0:
        return features
    
    duration = features["duration"]
    
    if features["has_audio"]:
        min_dur = duration
        sample_starts = []
        
        if min_dur > SKIP_FIRST_SECONDS + AUDIO_SAMPLE_DURATION:
            available = min_dur - SKIP_FIRST_SECONDS - AUDIO_SAMPLE_DURATION
            for i in range(NUM_AUDIO_SAMPLES):
                start = SKIP_FIRST_SECONDS + (available * i / max(NUM_AUDIO_SAMPLES - 1, 1))
                sample_starts.append(start)
        else:
            mid = min_dur / 2
            sample_starts = [max(0, mid - AUDIO_SAMPLE_DURATION/2)]
        
        for start in sample_starts:
            audio_data = extract_audio_sample(video_path, start, AUDIO_SAMPLE_DURATION)
            if audio_data is not None:
                fp = generate_audio_fingerprint(audio_data)
                if len(fp) > 0:
                    features["audio_fingerprints"].append(fp.tolist())
    
    # Batch extract visual samples
    images = extract_visual_samples_batch(video_path, duration, effective_temp_dir)
    
    if images:
        features["visual_hashes"] = generate_visual_fingerprint(images)
    
    if features.get("duration", 0) > 0:
        save_features_to_cache(features)
    
    return features


def calculate_frequency_score(image: np.ndarray) -> float:
    """
    Analyze frequency domain to detect missing high-frequency detail.
    Returns score 0-1, higher = more likely upscaled.
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Get dimensions
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Define frequency bands (concentric rings)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        
        # Calculate energy in different frequency bands
        low_freq_mask = x*x + y*y <= (min(h, w) * 0.1)**2
        mid_freq_mask = (x*x + y*y > (min(h, w) * 0.1)**2) & (x*x + y*y <= (min(h, w) * 0.3)**2)
        high_freq_mask = x*x + y*y > (min(h, w) * 0.3)**2
        
        low_energy = np.sum(magnitude[low_freq_mask])
        mid_energy = np.sum(magnitude[mid_freq_mask])
        high_energy = np.sum(magnitude[high_freq_mask])
        
        total_energy = low_energy + mid_energy + high_energy
        if total_energy == 0:
            return 0.0
        
        # Calculate ratios
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Natural video has significant high-frequency content
        # Upscaled video has low high-frequency energy
        # Score increases as high-frequency ratio decreases
        expected_high_freq = 0.15  # Expected minimum for genuine high-res
        
        if high_ratio >= expected_high_freq:
            return 0.0
        else:
            # Linear scale: 0 at expected threshold, 1 at 0 high freq
            return min(1.0, (expected_high_freq - high_ratio) / expected_high_freq)
            
    except Exception as e:
        return 0.0


def calculate_edge_sharpness_score(image: np.ndarray) -> float:
    """
    Analyze edge sharpness using Laplacian variance.
    Upscaled images tend to have lower sharpness.
    Returns score 0-1, higher = more likely upscaled.
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Apply Laplacian operator
        laplacian = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
        
        # Convolve
        laplacian_img = convolve(gray, laplacian)
        
        # Calculate variance (measure of sharpness)
        variance = np.var(laplacian_img)
        
        # Normalize by image size
        variance_normalized = variance / (gray.shape[0] * gray.shape[1])
        
        # Expected variance ranges (empirically determined)
        # These are conservative estimates to avoid flagging deliberately soft content
        expected_var_1080p = 0.5
        expected_var_4k = 1.5
        
        # Determine which threshold to use based on image size
        h, w = gray.shape
        if max(h, w) >= 3840:  # 4K
            expected = expected_var_4k
        elif max(h, w) >= 1920:  # 1080p
            expected = expected_var_1080p
        else:
            return 0.0
        
        # Score: 0 if variance meets expectation, increases as it falls short
        if variance_normalized >= expected * 0.6:  # 60% threshold for soft content
            return 0.0
        else:
            ratio = variance_normalized / (expected * 0.6)
            return max(0.0, min(1.0, 1.0 - ratio))
            
    except Exception as e:
        return 0.0


def calculate_multiscale_score(video_path: str, width: int, height: int) -> float:
    """
    Multi-scale analysis: downsample to lower res and compare.
    If quality is nearly identical when downsampled, likely upscaled.
    Returns score 0-1, higher = more likely upscaled.
    """
    try:
        if TEMP_DIR is None:
            return 0.0
        
        # Only analyze 1080p+ content
        if max(width, height) < 1920:
            return 0.0
        
        # Extract a frame at 50% mark
        duration = get_video_duration(video_path)
        timestamp = duration * 0.5
        
        temp_orig = os.path.join(TEMP_DIR, f"upscale_orig_{os.path.basename(video_path)}.jpg")
        temp_down = os.path.join(TEMP_DIR, f"upscale_down_{os.path.basename(video_path)}.jpg")
        
        # Extract original resolution frame
        cmd_orig = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
            '-vframes', '1', '-q:v', '2', temp_orig
        ]
        subprocess.run(cmd_orig, capture_output=True, timeout=15)
        
        if not os.path.exists(temp_orig):
            return 0.0
        
        # Extract downsampled frame (to 720p equivalent)
        target_width = 1280
        target_height = 720
        
        cmd_down = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}:flags=lanczos',
            '-vframes', '1', '-q:v', '2', temp_down
        ]
        subprocess.run(cmd_down, capture_output=True, timeout=15)
        
        if not os.path.exists(temp_down):
            os.remove(temp_orig)
            return 0.0
        
        # Load images
        img_orig = Image.open(temp_orig).convert('RGB')
        img_down = Image.open(temp_down).convert('RGB')
        
        # Resize original to match downsampled for comparison
        img_orig_resized = img_orig.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Calculate structural similarity
        arr_orig = np.array(img_orig_resized).astype(np.float32)
        arr_down = np.array(img_down).astype(np.float32)
        
        # Simple normalized cross-correlation
        mean_orig = np.mean(arr_orig)
        mean_down = np.mean(arr_down)
        
        std_orig = np.std(arr_orig)
        std_down = np.std(arr_down)
        
        if std_orig == 0 or std_down == 0:
            similarity = 1.0
        else:
            covariance = np.mean((arr_orig - mean_orig) * (arr_down - mean_down))
            similarity = covariance / (std_orig * std_down)
            similarity = max(0, min(1, (similarity + 1) / 2))  # Normalize to 0-1
        
        # Cleanup
        os.remove(temp_orig)
        os.remove(temp_down)
        
        # High similarity suggests upscaling (downsampling didn't lose much detail)
        # Threshold: if similarity > 0.95, likely upscaled
        if similarity > 0.95:
            return min(1.0, (similarity - 0.95) / 0.05)
        else:
            return 0.0
            
    except Exception as e:
        return 0.0


def detect_upscaling(video_path: str) -> Tuple[bool, float, Dict]:
    """
    Detect if a video is upscaled from lower resolution.
    Returns (is_upscaled, confidence_score, details_dict).
    Uses multiple methods with conservative thresholds to avoid false positives.
    """
    width, height = get_video_resolution(video_path)
    
    # Skip if resolution is too low to analyze
    if max(width, height) < MIN_UPSCALING_ANALYSIS_RESOLUTION:
        return False, 0.0, {"skipped": True, "reason": "resolution_too_low", "resolution": (int(width), int(height))}
    
    # Extract a representative frame for analysis
    duration = get_video_duration(video_path)
    timestamp = max(SKIP_FIRST_SECONDS, duration * 0.3)  # Avoid intro, not too late
    
    try:
        if TEMP_DIR is None:
            return False, 0.0, {"error": "temp_dir_not_set"}
        
        temp_frame = os.path.join(TEMP_DIR, f"upscale_analysis_{os.path.basename(video_path)}.jpg")
        
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
            '-vframes', '1', '-q:v', '2', temp_frame
        ]
        subprocess.run(cmd, capture_output=True, timeout=15)
        
        if not os.path.exists(temp_frame):
            return False, 0.0, {"error": "frame_extraction_failed"}
        
        # Load image
        img = Image.open(temp_frame).convert('RGB')
        img_array = np.array(img)
        
        # Run multiple detection methods
        freq_score = calculate_frequency_score(img_array)
        edge_score = calculate_edge_sharpness_score(img_array)
        
        # Cleanup
        os.remove(temp_frame)
        
        # Multi-scale test (separate from frame analysis)
        multi_score = calculate_multiscale_score(video_path, width, height)
        
        # Combine scores with weights
        # Frequency analysis is most reliable, but edge detection catches some cases
        # Multi-scale is slowest but most accurate when it works
        weights = {
            'frequency': 0.35,
            'edge': 0.25,
            'multiscale': 0.40
        }
        
        combined_score = (
            weights['frequency'] * freq_score +
            weights['edge'] * edge_score +
            weights['multiscale'] * multi_score
        )
        
        # Determine if upscaled (conservative threshold)
        is_upscaled = bool(combined_score > UPSCALING_CONFIDENCE_THRESHOLD)
        
        # Convert numpy types to native Python types for JSON serialization
        details = {
            "resolution": (int(width), int(height)),
            "scores": {
                "frequency": round(float(freq_score), 3),
                "edge_sharpness": round(float(edge_score), 3),
                "multiscale": round(float(multi_score), 3),
                "combined": round(float(combined_score), 3)
            },
            "threshold": float(UPSCALING_CONFIDENCE_THRESHOLD),
            "methods_used": ["frequency_analysis", "edge_sharpness", "multiscale_comparison"]
        }
        
        return is_upscaled, float(combined_score), details
        
    except Exception as e:
        return False, 0.0, {"error": str(e), "resolution": (int(width), int(height))}


def extract_video_metadata(video_path: str) -> Dict:
    """
    Extract comprehensive video metadata using ffprobe.
    Returns dictionary with all metadata fields.
    """
    metadata = {
        "file_info": {},
        "video": {},
        "audio": [],
        "container": {},
        "subtitles": {}
    }
    
    try:
        # File info
        file_stat = os.stat(video_path)
        metadata["file_info"] = {
            "original_full_path": os.path.abspath(video_path),
            "filename": os.path.basename(video_path),
            "file_size_bytes": file_stat.st_size,
            "modification_time": file_stat.st_mtime
        }
        
        # Get all streams info
        cmd = [
            'ffprobe', '-v', 'error', '-show_streams', '-show_format',
            '-print_format', 'json', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return metadata
        
        probe_data = json.loads(result.stdout)
        
        # Process streams
        video_stream = None
        audio_streams = []
        subtitle_streams = []
        
        for stream in probe_data.get('streams', []):
            codec_type = stream.get('codec_type', '')
            if codec_type == 'video':
                if not video_stream or stream.get('disposition', {}).get('default', 0):
                    video_stream = stream
            elif codec_type == 'audio':
                audio_streams.append(stream)
            elif codec_type == 'subtitle':
                subtitle_streams.append(stream)
        
        # Video metadata
        if video_stream:
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(video_stream.get('duration', 0) or 
                           probe_data.get('format', {}).get('duration', 0) or 0)
            
            # Format duration as HH:MM:SS
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Extract frame rate
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            except:
                fps = 0
            
            # Bit depth calculation
            pix_fmt = video_stream.get('pix_fmt', '')
            bit_depth = 10 if 'p10' in pix_fmt else 8
            
            # HDR detection
            hdr_formats = ['smpte2084', 'arib-std-b67', 'smpte2086', 'cll']
            color_transfer = video_stream.get('color_transfer', '').lower()
            color_space = video_stream.get('color_space', '')
            hdr_format = None
            if any(fmt in color_transfer for fmt in hdr_formats):
                hdr_format = 'HDR10' if '2084' in color_transfer else 'HLG' if 'arib' in color_transfer else 'HDR'
            
            # Video bitrate
            video_bitrate = 0
            if 'bit_rate' in video_stream:
                video_bitrate = int(video_stream['bit_rate']) // 1000
            
            metadata["video"] = {
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "duration_seconds": round(duration, 2),
                "duration_formatted": duration_formatted,
                "video_codec": video_stream.get('codec_name', 'unknown'),
                "video_bitrate_kbps": video_bitrate,
                "frame_rate": round(fps, 2),
                "pixel_format": pix_fmt,
                "video_profile": video_stream.get('profile', 'unknown'),
                "bit_depth": bit_depth,
                "color_space": color_space or 'unknown',
                "hdr_format": hdr_format
            }
        
        # Audio metadata
        for audio_stream in audio_streams:
            audio_info = {
                "codec": audio_stream.get('codec_name', 'unknown'),
                "bitrate_kbps": int(audio_stream.get('bit_rate', 0)) // 1000 if 'bit_rate' in audio_stream else 0,
                "sample_rate_hz": int(audio_stream.get('sample_rate', 0)),
                "channels": audio_stream.get('channels', 0),
                "language": audio_stream.get('tags', {}).get('language', 'und')
            }
            metadata["audio"].append(audio_info)
        
        # Container metadata
        format_info = probe_data.get('format', {})
        metadata["container"] = {
            "container_format": format_info.get('format_name', 'unknown').split(',')[0],
            "creation_time": format_info.get('tags', {}).get('creation_time', None)
        }
        
        # Subtitle metadata
        subtitle_languages = []
        for sub_stream in subtitle_streams:
            lang = sub_stream.get('tags', {}).get('language', 'und')
            subtitle_languages.append(lang)
        
        metadata["subtitles"] = {
            "subtitle_tracks_count": len(subtitle_streams),
            "subtitle_languages": subtitle_languages
        }
        
    except Exception as e:
        print(f"Warning: Could not extract metadata for {video_path}: {e}")
    
    return metadata


def calculate_quality_score(metadata: Dict) -> Tuple[int, str]:
    """
    Calculate quality score (0-100) based on video characteristics.
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
    
    # 1. Resolution scoring - proportional to pixel count (base is 1920x1080)
    max_1080p_pixels = 1920 * 1080
    resolution_score = int(40 * resolution_pixels / max_1080p_pixels)
    score += resolution_score
    if resolution_pixels >= 3840 * 2160:
        reasons.append(f"4K resolution ({width}x{height})")
    elif resolution_pixels >= 1920 * 1080:
        reasons.append(f"1080p resolution ({width}x{height})")
    elif resolution_pixels >= 1280 * 720:
        reasons.append(f"720p resolution ({width}x{height})")
    else:
        reasons.append(f"resolution ({width}x{height})")
    
    # 2. Resolution tier bonuses - prioritize higher resolution
    pixels_720p = 1280 * 720  # 921,600 pixels
    pixels_1080p = 1920 * 1080  # 2,073,600 pixels
    
    if resolution_pixels >= pixels_1080p:
        score += 25
        reasons.append(f"1080p+ resolution ({width}x{height})")
    elif resolution_pixels >= pixels_720p:
        score += 15
        reasons.append(f"720p+ resolution ({width}x{height})")
    else:
        reasons.append(f"below 720p resolution ({width}x{height})")
    
    # 3. Bitrate scoring (20 points max) - linear per-pixel minimum bitrate
    bitrate = video_info.get("video_bitrate_kbps", 0)
    if resolution_pixels > 0:
        # Codec efficiency multipliers (relative to H.264 baseline)
        CODEC_EFFICIENCY = {
            'av1': 3.0,      # AV1 is ~3x more efficient than H.264
            'hevc': 2.0,     # HEVC/H.265 is ~2x more efficient
            'h265': 2.0,
            'h264': 1.0,     # Baseline
            'avc': 1.0,
        }
        
        # Normalize bitrate by codec efficiency
        codec = video_info.get("video_codec", "unknown").lower()
        efficiency_multiplier = CODEC_EFFICIENCY.get(codec, 1.0)
        effective_bitrate = int(bitrate * efficiency_multiplier)
        
        # Calculate optimal bitrate: 2000 kbps per megapixel
        megapixels = resolution_pixels / 1_000_000
        optimal_bitrate = megapixels * 2000  # kbps
        
        # Calculate bitrate score - continuous scoring with diminishing returns above optimal
        # Reduced maximum from 30 to 20 points to balance with increased fps weighting
        if effective_bitrate > 0 and optimal_bitrate > 0:
            ratio = effective_bitrate / optimal_bitrate
            
            if ratio < 0.3:
                # Below 30% of optimal: linear penalty
                bitrate_score = int(13 * (ratio / 0.3))  # was 20, now ~13
                bitrate_quality = "below optimal"
            elif ratio < 1.0:
                # Between 30% and 100%: linear scaling from 5 to 13 points
                # This ensures smooth transition to the "above optimal" formula
                bitrate_score = int(5 + 8 * (ratio - 0.3) / 0.7)  # was 8+12*(...), now 5+8*(...)
                bitrate_quality = "approaching optimal"
            else:
                # Above 100%: diminishing returns but still increasing
                # sqrt curve: score = 13 * sqrt(ratio), max 20 points
                bitrate_score = min(20, int(13 * (ratio ** 0.5)))  # was min(30, 20*(...)), now min(20, 13*(...))
                bitrate_quality = "above optimal"
        else:
            bitrate_score = 0
            bitrate_quality = "unknown"
        
        score += bitrate_score
        
        # Report bitrate with ratio to optimal
        if optimal_bitrate > 0:
            ratio = effective_bitrate / optimal_bitrate if effective_bitrate > 0 else 0
            if efficiency_multiplier > 1.0:
                reasons.append(f"{bitrate_quality} effective bitrate ({bitrate} kbps × {efficiency_multiplier:.0f} = {effective_bitrate} kbps, {ratio:.2f}x optimal)")
            else:
                reasons.append(f"{bitrate_quality} bitrate ({effective_bitrate} kbps, {ratio:.2f}x optimal)")
        elif bitrate > 0:
            reasons.append(f"moderate bitrate ({bitrate} kbps)")
        else:
            reasons.append("unknown bitrate")
    
    # 4. Codec efficiency (20 points max)
    codec = video_info.get("video_codec", "unknown").lower()
    if codec in ['hevc', 'h265', 'av1']:
        score += 20
        reasons.append("efficient modern codec (HEVC/AV1)")
    elif codec in ['h264', 'avc']:
        score += 15
        reasons.append("standard H.264 codec")
    elif codec in ['mpeg4', 'mpeg2', 'xvid', 'divx']:
        score += 8
        reasons.append("older MPEG codec")
    else:
        score += 5
        reasons.append(f"codec: {codec}")
    
    # 5. HDR bonus (15 points)
    hdr = video_info.get("hdr_format")
    if hdr:
        score += 15
        reasons.append(f"HDR support ({hdr})")
    
    # 6. Frame rate (20 points max - increased priority)
    fps = video_info.get("frame_rate", 0)
    if fps >= 59:  # 60fps
        score += 20
        reasons.append("60fps (high priority)")
    elif fps >= 29:  # 30fps
        score += 14
        reasons.append("30fps (high priority)")
    elif fps >= 24:
        score += 10
        reasons.append("24/25fps (high priority)")
    else:
        score += 6
        reasons.append(f"{fps}fps (low priority)")
    
    reason_str = ", ".join(reasons)
    return max(0, score), reason_str


def analyze_duplicate_set(videos_with_metadata: List[Tuple[str, Dict]]) -> Dict[str, Dict]:
    """
    Analyze a set of duplicate videos and determine which to keep.
    Returns dictionary mapping video paths to analysis results.
    """
    if not videos_with_metadata:
        return {}
    
    # Calculate quality scores for all videos
    results = {}
    for video_path, metadata in videos_with_metadata:
        score, reason = calculate_quality_score(metadata)
        results[video_path] = {
            "metadata": metadata,
            "quality_score": score,
            "quality_reason": reason,
            "recommendation": "DELETE",
            "reason": "",
            "better_alternative": None
        }
    
    # Find the highest quality video
    if len(results) == 1:
        # Single video - keep it
        video_path = list(results.keys())[0]
        results[video_path]["recommendation"] = "KEEP"
        results[video_path]["reason"] = "Only copy of this content"
        return results
    
    # Find best video based on quality score, tie-break by modification time, width and bitrate
    # Prefer older files (lower modification_time) when scores are within 1 point
    best_path = max(results.keys(), key=lambda p: (
        results[p]["quality_score"],
        -results[p]["metadata"].get("file_info", {}).get("modification_time", 0),
        results[p]["metadata"].get("video", {}).get("width", 0),
        results[p]["metadata"].get("video", {}).get("video_bitrate_kbps", 0)
    ))
    best_score = results[best_path]["quality_score"]
    best_metadata = results[best_path]["metadata"]
    
    # Mark the best as KEEP
    best_video_info = best_metadata.get("video", {})
    best_res = best_video_info.get("resolution", "unknown")
    best_codec = best_video_info.get("video_codec", "unknown")
    best_bitrate = best_video_info.get("video_bitrate_kbps", 0)
    best_mtime = best_metadata.get("file_info", {}).get("modification_time", 0)
    
    # Check if best won by modification time tie-breaker
    other_scores = [results[p]["quality_score"] for p in results if p != best_path]
    won_by_tiebreaker = other_scores and best_score in other_scores and max(other_scores) == best_score
    
    # Mark the best as KEEP
    results[best_path]["recommendation"] = "KEEP"
    if won_by_tiebreaker:
        results[best_path]["reason"] = f"Best quality: {best_res}, {best_codec}, {results[best_path]['quality_score']} score (older file kept)"
    else:
        results[best_path]["reason"] = f"Best quality: {best_res}, {best_codec}, {results[best_path]['quality_score']} score"
    
    # Mark others as DELETE with reference to best
    for video_path, result in results.items():
        if video_path != best_path:
            score = result["quality_score"]
            score_diff = best_score - score
            
            video_info = result["metadata"].get("video", {})
            video_res = video_info.get("resolution", "unknown")
            video_codec = video_info.get("video_codec", "unknown")
            video_bitrate = video_info.get("video_bitrate_kbps", 0)
            
            # Generate reason with tie-breaker info
            if score_diff >= 30:
                reason = f"Significantly lower quality ({score} vs {best_score})"
            elif score_diff >= 15:
                reason = f"Lower quality ({score} vs {best_score})"
            elif score_diff >= 1:
                reason = f"Lower quality ({score} vs {best_score})"
            else:
                # Scores are equal or within 1 point - check modification time
                result_mtime = result["metadata"].get("file_info", {}).get("modification_time", 0)
                if result_mtime > best_mtime:
                    reason = f"Same quality ({score} vs {best_score}) - kept older file"
                else:
                    reason = f"Same quality ({score} vs {best_score})"
            
            # Add specific technical reasons
            video_width = video_info.get("width", 0)
            video_height = video_info.get("height", 0)
            best_width = best_video_info.get("width", 0)
            best_height = best_video_info.get("height", 0)
            video_pixels = video_width * video_height
            best_pixels = best_width * best_height
            if video_pixels < best_pixels:
                reason += f"; lower resolution ({video_width}x{video_height} vs {best_width}x{best_height})"
            if video_codec != best_codec and best_codec in ['hevc', 'h265', 'av1']:
                reason += f"; less efficient codec ({video_codec} vs {best_codec})"
            if video_bitrate < best_bitrate * 0.7 and video_bitrate > 0:
                reason += f"; lower bitrate ({video_bitrate} vs {best_bitrate} kbps)"
            
            result["recommendation"] = "DELETE"
            result["reason"] = reason
            result["better_alternative"] = {
                "filename": os.path.basename(best_path),
                "quality_score": best_score,
                "reason": results[best_path]["reason"]
            }
    
    return results


def organize_duplicates(directory: str, duplicate_groups: List[List[str]], dry_run: bool = False, create_markers: bool = False):
    """Move duplicate videos into __deduped/numbered folders with metadata JSON files."""
    if not duplicate_groups:
        print("No duplicates found!")
        return
    
    print(f"\nFound {len(duplicate_groups)} duplicate sets")
    
    # Create __deduped folder
    deduped_base = os.path.join(directory, "__deduped")
    
    for i, group in enumerate(duplicate_groups, 1):
        folder_name = f"duplicate_set_{i:03d}"
        folder_path = os.path.join(deduped_base, folder_name)
        
        print(f"\n{folder_name}: {len(group)} videos")
        
        # Extract metadata and calculate quality scores for all videos in this set
        videos_with_metadata = []
        for video_path in group:
            print(f"  Extracting metadata: {os.path.basename(video_path)}")
            metadata = extract_video_metadata(video_path)
            videos_with_metadata.append((video_path, metadata))
        
        # Analyze the duplicate set to determine recommendations
        analysis_results = analyze_duplicate_set(videos_with_metadata)
        
        for video_path in group:
            rel_path = os.path.relpath(video_path, directory)
            analysis = analysis_results.get(video_path, {})
            rec = analysis.get("recommendation", "UNKNOWN")
            score = analysis.get("quality_score", 0)
            print(f"  - {rel_path} [{rec}, score: {score}]")
        
        if not dry_run:
            # Create __deduped folder and set folder
            os.makedirs(folder_path, exist_ok=True)
            
            # Move all videos in the group and create metadata JSON files
            for video_path in group:
                try:
                    filename = os.path.basename(video_path)
                    dest_path = os.path.join(folder_path, filename)
                    
                    # Handle name collisions
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(folder_path, f"{base_name}_{counter:02d}{ext}")
                        counter += 1
                    
                    # Get the analysis result for this video
                    analysis = analysis_results.get(video_path, {})
                    metadata = analysis.get("metadata", {})
                    
                    # Build the full JSON structure
                    json_data = {
                        "recommendation": analysis.get("recommendation", "UNKNOWN"),
                        "original_full_path": analysis.get("metadata", {}).get("file_info", {}).get("original_full_path", os.path.abspath(video_path)),
                        "quality_score": analysis.get("quality_score", 0),
                        "reason": analysis.get("reason", ""),
                        "filename": filename,
                        "file_size_bytes": metadata.get("file_info", {}).get("file_size_bytes", 0),
                        "modification_time": metadata.get("file_info", {}).get("modification_time", 0),
                        **metadata
                    }
                    
                    # Add better_alternative for DELETE files
                    if analysis.get("recommendation") == "DELETE":
                        json_data["better_alternative"] = analysis.get("better_alternative")
                    
                    # Save metadata JSON file (video.mp4.json)
                    json_filename = f"{os.path.basename(dest_path)}.json"
                    json_path = os.path.join(folder_path, json_filename)
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    
                    # Create marker file (.keep or .delete) only if requested
                    if create_markers:
                        recommendation = analysis.get("recommendation", "")
                        if recommendation == "KEEP":
                            marker_filename = f"{os.path.basename(dest_path)}.keep"
                            marker_path = os.path.join(folder_path, marker_filename)
                            open(marker_path, 'w').close()
                            print(f"  -> Created marker: {marker_filename}")
                        elif recommendation == "DELETE":
                            marker_filename = f"{os.path.basename(dest_path)}.delete"
                            marker_path = os.path.join(folder_path, marker_filename)
                            open(marker_path, 'w').close()
                            print(f"  -> Created marker: {marker_filename}")
                    
                    # Move the video file
                    shutil.move(video_path, dest_path)
                    print(f"  -> Moved to __deduped/{folder_name}")
                    print(f"  -> Created metadata: {json_filename}")
                    
                except Exception as e:
                    print(f"  ERROR moving {video_path}: {e}")


def main():
    global TEMP_DIR
    
    parser = argparse.ArgumentParser(description='Find and organize duplicate videos')
    parser.add_argument('directory', help='Directory containing videos')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show duplicates without moving files')
    parser.add_argument('--report', type=str, default='duplicates_report.json',
                        help='Path to save JSON report')
    parser.add_argument('--detect-upscaling', action='store_true',
                        help='Detect videos that are upscaled from lower resolutions (e.g., 720p encoded as 4K). Adds upscaling analysis to the report.')
    parser.add_argument('--create-markers', action='store_true',
                        help='Create marker files (.keep/.delete) for duplicate videos in __deduped folders')
    parser.add_argument('--include-subfolders', nargs='*', metavar='PATH',
                        help='Include subfolders in analysis. Without arguments: includes all subfolders (default). With arguments: includes only specified subfolder paths (relative to directory).')
    parser.add_argument('--no-subfolders', action='store_true',
                        help='Only scan the root directory (default is to include all subfolders)')
    parser.add_argument('--include-hidden', action='store_true',
                        help='Include hidden subfolders (starting with .). Default is to exclude them.')
    parser.add_argument('--exclude-root', action='store_true',
                        help='Exclude videos from the root directory. Only useful with --include-subfolders.')
    parser.add_argument('--visual-sim', type=float, default=None,
                        help='Minimum visual similarity threshold (0.0-1.0). Lower = more lenient. Default: 0.7')
    parser.add_argument('--visual-samples', type=int, default=None,
                        help='Number of frames to extract per video for visual comparison (1-10). Default: 4')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show debug output for comparison stages (useful for debugging why duplicates are not detected)')
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.directory)
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    # Create temp directory
    TEMP_DIR = tempfile.mkdtemp(prefix="video_dedup_")
    
    upscaling_results = {}
    
    try:
        # Determine scanning parameters
        # Default: include all subfolders, unless --no-subfolders is set
        if args.no_subfolders:
            include_subfolders = None  # Root only
        elif args.include_subfolders is not None:
            include_subfolders = args.include_subfolders  # Use specified value
        else:
            include_subfolders = []  # All subfolders (new default)
        
        exclude_root = args.exclude_root
        
        # Print scanning info
        if include_subfolders is None:
            print(f"Scanning {directory} (root directory only)...")
        elif len(include_subfolders) == 0:
            print(f"Scanning {directory} and all subfolders...")
        else:
            print(f"Scanning {directory} with specific subfolders:")
            for path in include_subfolders:
                print(f"  - {path}")
        
        if exclude_root:
            print("  (excluding root directory)")
        
        print(f"Using {HASH_METHOD} for file hashing")
        
        include_hidden = args.include_hidden
        
        # Apply command-line overrides for visual settings
        if args.visual_sim is not None:
            global VISUAL_SIM_THRESHOLD
            VISUAL_SIM_THRESHOLD = max(0.0, min(1.0, args.visual_sim))
            print(f"Visual similarity threshold: {VISUAL_SIM_THRESHOLD}")
        
        if args.visual_samples is not None:
            global NUM_VISUAL_SAMPLES
            NUM_VISUAL_SAMPLES = max(1, min(10, args.visual_samples))
            print(f"Visual samples: {NUM_VISUAL_SAMPLES}")
        
        # Find all videos
        video_paths = find_videos(directory, include_subfolders=include_subfolders, exclude_root=exclude_root, include_hidden=include_hidden)
        
        if not video_paths:
            print("No videos found!")
            sys.exit(0)
        
        # Sort by file size for better disk access pattern (SSD-friendly batching)
        video_paths = sorted(video_paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
        
        print(f"Found {len(video_paths)} videos")
        
        # Optional: Detect upscaling
        if args.detect_upscaling:
            print("\n" + "="*60)
            print("UPSCALING DETECTION MODE")
            print("="*60)
            print("Analyzing videos for potential upscaling...")
            print(f"Analyzing videos with resolution >= {MIN_UPSCALING_ANALYSIS_RESOLUTION}p")
            print(f"Confidence threshold: {UPSCALING_CONFIDENCE_THRESHOLD} (conservative)")
            print("Note: Deliberately soft content may not be flagged")
            print("="*60 + "\n")
            
            for path in tqdm(video_paths, desc="Analyzing upscaling"):
                is_upscaled, confidence, details = detect_upscaling(path)
                upscaling_results[path] = {
                    "is_upscaled": is_upscaled,
                    "confidence": round(confidence, 3),
                    "details": details
                }
            
            # Print summary
            upscaled_count = sum(1 for r in upscaling_results.values() if r["is_upscaled"])
            print(f"\nUpscaling Analysis Summary:")
            print(f"  Total videos analyzed: {len(upscaling_results)}")
            print(f"  Potentially upscaled: {upscaled_count}")
            
            if upscaled_count > 0:
                print(f"\n  Flagged videos:")
                for path, result in upscaling_results.items():
                    if result["is_upscaled"]:
                        width, height = result["details"].get("resolution", (0, 0))
                        print(f"    - {os.path.basename(path)} ({width}x{height}) [confidence: {result['confidence']:.2f}]")
        
        # Extract all features for each video (parallel)
        n_workers = os.cpu_count() or 4
        print(f"Using {n_workers} workers for feature extraction")
        
        features_list = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_all_features, path): path for path in video_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                try:
                    features = future.result()
                    if features.get("duration", 0) > 0:
                        features_list.append(features)
                except Exception as e:
                    pass
        
        if len(features_list) < 2:
            print("Not enough valid videos to compare")
            sys.exit(0)
        
        # Find duplicates using precomputed features
        duplicate_groups = find_duplicate_groups_with_features(features_list, verbose=args.verbose)
        
        # Save report
        report = {
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
                            'duration': next((f.get("duration", 0) for f in features_list if f["path"] == v), 0)
                        }
                        for v in group
                    ]
                }
                for i, group in enumerate(duplicate_groups)
            ]
        }
        
        # Add upscaling analysis to report if performed
        if args.detect_upscaling and upscaling_results:
            report['upscaling_analysis'] = {
                'enabled': True,
                'threshold': UPSCALING_CONFIDENCE_THRESHOLD,
                'min_resolution_analyzed': MIN_UPSCALING_ANALYSIS_RESOLUTION,
                'summary': {
                    'total_analyzed': len(upscaling_results),
                    'upscaled_detected': sum(1 for r in upscaling_results.values() if r['is_upscaled']),
                    'analysis_methods': ['frequency_analysis', 'edge_sharpness', 'multiscale_comparison']
                },
                'videos': [
                    {
                        'path': path,
                        'filename': os.path.basename(path),
                        'is_upscaled': result['is_upscaled'],
                        'confidence': result['confidence'],
                        'resolution': result['details'].get('resolution', [0, 0]),
                        'scores': result['details'].get('scores', {})
                    }
                    for path, result in upscaling_results.items()
                ]
            }
        
        report_path = os.path.join(directory, args.report)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        # Organize duplicates
        organize_duplicates(directory, duplicate_groups, args.dry_run, args.create_markers)
        
    finally:
        # Cleanup temp directory
        if TEMP_DIR and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


if __name__ == '__main__':
    main()
