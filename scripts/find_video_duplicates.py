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
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
import subprocess

import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import audioread
from scipy.ndimage import convolve

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov'}

# Configuration
LENGTH_TOLERANCE = 0.15  # 15% length difference allowed
AUDIO_SAMPLE_DURATION = 5  # seconds per audio sample
NUM_AUDIO_SAMPLES = 4  # number of samples to extract
NUM_VISUAL_SAMPLES = 4  # number of frames to extract
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


def extract_visual_samples(video_path: str, duration: float) -> List[Image.Image]:
    """Extract frame samples from video at specific timestamps."""
    if TEMP_DIR is None:
        return []
    
    images = []
    
    # Sample from 20%, 40%, 60%, 80% of duration (avoid beginning and end)
    sample_points = [0.2, 0.4, 0.6, 0.8]
    
    for point in sample_points[:NUM_VISUAL_SAMPLES]:
        timestamp = duration * point
        
        try:
            # Extract frame using ffmpeg
            temp_frame = os.path.join(TEMP_DIR, f"frame_{os.path.basename(video_path)}_{point}.jpg")
            
            cmd = [
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-q:v', '2', temp_frame
            ]
            subprocess.run(cmd, capture_output=True, timeout=15)
            
            if os.path.exists(temp_frame):
                img = Image.open(temp_frame)
                images.append(img)
                os.remove(temp_frame)
        except Exception as e:
            pass
    
    return images


def generate_visual_fingerprint(images: List[Image.Image]) -> List[str]:
    """Generate perceptual hashes for images."""
    hashes = []
    for img in images:
        try:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Use pHash (perceptual hash)
            hash_val = str(imagehash.phash(img))
            hashes.append(hash_val)
        except Exception as e:
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


def compare_visual_fingerprints(hashes1: List[str], hashes2: List[str]) -> float:
    """Compare visual fingerprints and return similarity (0-1)."""
    if not hashes1 or not hashes2:
        return 0.0
    
    # Compare all pairs and find best matches
    matches = 0
    threshold = 10  # Hamming distance threshold for match
    
    used = set()
    for h1 in hashes1:
        best_match = None
        best_dist = float('inf')
        
        for i, h2 in enumerate(hashes2):
            if i in used:
                continue
            dist = hamming_distance(h1, h2)
            if dist < best_dist:
                best_dist = dist
                best_match = i
        
        if best_match is not None and best_dist <= threshold:
            matches += 1
            used.add(best_match)
    
    # Return proportion of matches
    return matches / max(len(hashes1), len(hashes2))


def find_videos(
    directory: str,
    include_subfolders: Optional[List[str]] = None,
    exclude_root: bool = False
) -> List[str]:
    """
    Find all video files in directory.
    
    Args:
        directory: Base directory to scan
        include_subfolders: None = no subfolders, empty list = all subfolders (except .deduped),
                           list of paths = only those specific subfolders
        exclude_root: If True, don't include videos from the root directory
    
    Returns:
        List of video file paths
    """
    videos = []
    abs_directory = os.path.abspath(directory)
    
    # Always exclude the .deduped folder
    deduped_folder = os.path.join(abs_directory, ".deduped")
    
    def is_in_deduped(path: str) -> bool:
        """Check if path is inside the .deduped folder."""
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
            # Include all subfolders recursively, but exclude .deduped
            for root, _, files in os.walk(abs_directory):
                # Skip if this directory is inside .deduped
                if is_in_deduped(root):
                    continue
                
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
                
                # Skip if it's the .deduped folder
                if is_in_deduped(subfolder_path):
                    print(f"Warning: Skipping .deduped folder: {subfolder}")
                    continue
                
                folders_to_scan.append(subfolder_path)
            
            # Scan each folder recursively
            for folder in folders_to_scan:
                for root, _, files in os.walk(folder):
                    # Skip if this directory is inside .deduped
                    if is_in_deduped(root):
                        continue
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        ext = Path(file).suffix.lower()
                        if ext in VIDEO_EXTENSIONS:
                            videos.append(file_path)
    
    return videos


def is_same_video(video1: Tuple[str, float], video2: Tuple[str, float]) -> Tuple[bool, str]:
    """
    Check if two videos are the same content.
    Returns (is_duplicate, method_used).
    """
    path1, dur1 = video1
    path2, dur2 = video2
    
    # Check duration similarity
    duration_diff = abs(dur1 - dur2) / max(dur1, dur2, 1)
    if duration_diff > LENGTH_TOLERANCE:
        return False, "duration_mismatch"
    
    # Check if both have audio
    audio1 = has_audio_stream(path1)
    audio2 = has_audio_stream(path2)
    
    # Try audio fingerprinting if both have audio
    if audio1 and audio2:
        samples_match = 0
        total_samples = 0
        
        # Sample from multiple points (avoiding first 10 seconds)
        sample_starts = []
        min_dur = min(dur1, dur2)
        
        if min_dur > SKIP_FIRST_SECONDS + AUDIO_SAMPLE_DURATION:
            # Create sample points after the skip
            available = min_dur - SKIP_FIRST_SECONDS - AUDIO_SAMPLE_DURATION
            for i in range(NUM_AUDIO_SAMPLES):
                start = SKIP_FIRST_SECONDS + (available * i / max(NUM_AUDIO_SAMPLES - 1, 1))
                sample_starts.append(start)
        else:
            # Video is short, sample from middle
            mid = min_dur / 2
            sample_starts = [max(0, mid - AUDIO_SAMPLE_DURATION/2)]
        
        for start in sample_starts:
            audio_data1 = extract_audio_sample(path1, start, AUDIO_SAMPLE_DURATION)
            audio_data2 = extract_audio_sample(path2, start, AUDIO_SAMPLE_DURATION)
            
            if audio_data1 is not None and audio_data2 is not None:
                total_samples += 1
                fp1 = generate_audio_fingerprint(audio_data1)
                fp2 = generate_audio_fingerprint(audio_data2)
                
                similarity = compare_audio_fingerprints(fp1, fp2)
                if similarity > 0.90:  # Increased threshold for stricter matching
                    samples_match += 1
        
        # Require ALL samples to match (not just half) for audio fingerprint match
        if total_samples > 0 and samples_match >= total_samples:
            return True, "audio_fingerprint"
    
    # Fallback to visual fingerprinting
    images1 = extract_visual_samples(path1, dur1)
    images2 = extract_visual_samples(path2, dur2)
    
    if len(images1) >= 2 and len(images2) >= 2:
        hashes1 = generate_visual_fingerprint(images1)
        hashes2 = generate_visual_fingerprint(images2)
        
        visual_sim = compare_visual_fingerprints(hashes1, hashes2)
        if visual_sim >= 0.5:  # At least half the frames match
            return True, "visual_fingerprint"
    
    return False, "no_match"


def find_duplicate_groups(videos: List[Tuple[str, float]]) -> List[List[str]]:
    """Find groups of duplicate videos."""
    n = len(videos)
    if n == 0:
        return []
    
    # Group by approximate duration for efficiency
    duration_groups = defaultdict(list)
    for video in videos:
        # Round duration to nearest 30 seconds for grouping
        bucket = round(video[1] / 30) * 30
        duration_groups[bucket].append(video)
    
    # Find matches
    matches = defaultdict(set)
    
    print(f"Analyzing {n} videos for duplicates...")
    
    for bucket, bucket_videos in duration_groups.items():
        bucket_size = len(bucket_videos)
        if bucket_size < 2:
            continue
        
        # Compare all pairs in bucket
        for i in tqdm(range(bucket_size), desc=f"Bucket {bucket}s"):
            for j in range(i + 1, bucket_size):
                v1 = bucket_videos[i]
                v2 = bucket_videos[j]
                
                is_dup, method = is_same_video(v1, v2)
                if is_dup:
                    matches[v1[0]].add(v2[0])
                    matches[v2[0]].add(v1[0])
    
    # Build connected components (transitive matches)
    visited = set()
    groups = []
    
    for video in videos:
        video_path = video[0]
        if video_path in visited:
            continue
        
        # BFS to find all connected videos
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
    resolution_score = int(25 * resolution_pixels / max_1080p_pixels)
    score += resolution_score
    if resolution_pixels >= 3840 * 2160:
        reasons.append(f"4K resolution ({width}x{height})")
    elif resolution_pixels >= 1920 * 1080:
        reasons.append(f"1080p resolution ({width}x{height})")
    elif resolution_pixels >= 1280 * 720:
        reasons.append(f"720p resolution ({width}x{height})")
    else:
        reasons.append(f"resolution ({width}x{height})")
    
    # 2. Bitrate scoring (30 points max) - scaled to resolution and codec efficiency
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
        
        # Calculate expected bitrate for this resolution (very rough estimates)
        expected_4k_bitrate = 25000
        expected_1080p_bitrate = 8000
        expected_720p_bitrate = 4000
        
        # Use height-based tiers for bitrate comparison (consistent with resolution scoring)
        bitrate_ratio = min(1.0, effective_bitrate / expected_4k_bitrate) if height >= 2160 else \
                       min(1.0, effective_bitrate / expected_1080p_bitrate) if height >= 1080 else \
                       min(1.0, effective_bitrate / expected_720p_bitrate) if height >= 720 else \
                       min(1.0, effective_bitrate / 2000)
        
        bitrate_score = int(30 * bitrate_ratio)
        score += bitrate_score
        
        # Report effective bitrate for modern codecs
        if efficiency_multiplier > 1.0:
            if bitrate_score >= 25:
                reasons.append(f"excellent effective bitrate ({bitrate} kbps × {efficiency_multiplier:.0f} = {effective_bitrate} kbps)")
            elif bitrate_score >= 15:
                reasons.append(f"good effective bitrate ({bitrate} kbps × {efficiency_multiplier:.0f} = {effective_bitrate} kbps)")
            elif bitrate > 0:
                reasons.append(f"moderate effective bitrate ({bitrate} kbps × {efficiency_multiplier:.0f} = {effective_bitrate} kbps)")
            else:
                reasons.append("unknown bitrate")
        else:
            if bitrate_score >= 25:
                reasons.append(f"excellent bitrate ({bitrate} kbps)")
            elif bitrate_score >= 15:
                reasons.append(f"good bitrate ({bitrate} kbps)")
            elif bitrate > 0:
                reasons.append(f"moderate bitrate ({bitrate} kbps)")
            else:
                reasons.append("unknown bitrate")
    
    # 3. Codec efficiency (20 points max)
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
    
    # 4. HDR bonus (15 points)
    hdr = video_info.get("hdr_format")
    if hdr:
        score += 15
        reasons.append(f"HDR support ({hdr})")
    
    # 5. Frame rate (10 points max)
    fps = video_info.get("frame_rate", 0)
    if fps >= 59:  # 60fps
        score += 10
        reasons.append("60fps")
    elif fps >= 29:  # 30fps
        score += 7
        reasons.append("30fps")
    elif fps >= 24:
        score += 5
        reasons.append("24/25fps")
    else:
        score += 3
        reasons.append(f"{fps}fps")
    
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
            "recommendation": "DELETE_CANDIDATE",
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
    
    # Find best video based on quality score, tie-break by width and bitrate
    best_path = max(results.keys(), key=lambda p: (results[p]["quality_score"], results[p]["metadata"].get("video", {}).get("width", 0), results[p]["metadata"].get("video", {}).get("video_bitrate_kbps", 0)))
    best_score = results[best_path]["quality_score"]
    best_metadata = results[best_path]["metadata"]
    
    # Mark the best as KEEP
    best_video_info = best_metadata.get("video", {})
    best_res = best_video_info.get("resolution", "unknown")
    best_codec = best_video_info.get("video_codec", "unknown")
    best_bitrate = best_video_info.get("video_bitrate_kbps", 0)
    
    results[best_path]["recommendation"] = "KEEP"
    results[best_path]["reason"] = f"Best quality: {best_res}, {best_codec}, {results[best_path]['quality_score']} score"
    
    # Mark others as DELETE_CANDIDATE with reference to best
    for video_path, result in results.items():
        if video_path != best_path:
            score = result["quality_score"]
            score_diff = best_score - score
            
            video_info = result["metadata"].get("video", {})
            video_res = video_info.get("resolution", "unknown")
            video_codec = video_info.get("video_codec", "unknown")
            video_bitrate = video_info.get("video_bitrate_kbps", 0)
            
            # Generate reason
            if score_diff >= 30:
                reason = f"Significantly lower quality ({score} vs {best_score})"
            elif score_diff >= 15:
                reason = f"Lower quality ({score} vs {best_score})"
            else:
                reason = f"Lower quality ({score} vs {best_score})"
            
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
            
            result["recommendation"] = "DELETE_CANDIDATE"
            result["reason"] = reason
            result["better_alternative"] = {
                "filename": os.path.basename(best_path),
                "quality_score": best_score,
                "reason": f"Better quality: {best_res}, {best_codec}, {best_score} score"
            }
    
    return results


def organize_duplicates(directory: str, duplicate_groups: List[List[str]], dry_run: bool = False):
    """Move duplicate videos into .deduped/numbered folders with metadata JSON files."""
    if not duplicate_groups:
        print("No duplicates found!")
        return
    
    print(f"\nFound {len(duplicate_groups)} duplicate sets")
    
    # Create .deduped folder
    deduped_base = os.path.join(directory, ".deduped")
    
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
            # Create .deduped folder and set folder
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
                    
                    # Add better_alternative for DELETE_CANDIDATE files
                    if analysis.get("recommendation") == "DELETE_CANDIDATE":
                        json_data["better_alternative"] = analysis.get("better_alternative")
                    
                    # Save metadata JSON file (video.mp4.json)
                    json_filename = f"{os.path.basename(dest_path)}.json"
                    json_path = os.path.join(folder_path, json_filename)
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    
                    # Create marker file (.keep or .delete)
                    recommendation = analysis.get("recommendation", "")
                    if recommendation == "KEEP":
                        marker_filename = f"{os.path.basename(dest_path)}.keep"
                        marker_path = os.path.join(folder_path, marker_filename)
                        open(marker_path, 'w').close()
                        print(f"  -> Created marker: {marker_filename}")
                    elif recommendation == "DELETE_CANDIDATE":
                        marker_filename = f"{os.path.basename(dest_path)}.delete"
                        marker_path = os.path.join(folder_path, marker_filename)
                        open(marker_path, 'w').close()
                        print(f"  -> Created marker: {marker_filename}")
                    
                    # Move the video file
                    shutil.move(video_path, dest_path)
                    print(f"  -> Moved to .deduped/{folder_name}")
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
    parser.add_argument('--include-subfolders', nargs='*', metavar='PATH',
                        help='Include subfolders in analysis. Without arguments: includes all subfolders (except .deduped). With arguments: includes only specified subfolder paths (relative to directory).')
    parser.add_argument('--exclude-root', action='store_true',
                        help='Exclude videos from the root directory. Only useful with --include-subfolders.')
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.directory)
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    # Validate argument combinations
    if args.exclude_root and args.include_subfolders is None:
        print("Error: --exclude-root requires --include-subfolders to be specified")
        sys.exit(1)
    
    # Create temp directory
    TEMP_DIR = tempfile.mkdtemp(prefix="video_dedup_")
    
    upscaling_results = {}
    
    try:
        # Determine scanning parameters
        include_subfolders = args.include_subfolders
        exclude_root = args.exclude_root
        
        # Print scanning info
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
        
        # Find all videos
        video_paths = find_videos(directory, include_subfolders=include_subfolders, exclude_root=exclude_root)
        
        if not video_paths:
            print("No videos found!")
            sys.exit(0)
        
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
        
        # Get durations
        videos_with_duration = []
        for path in tqdm(video_paths, desc="Getting durations"):
            dur = get_video_duration(path)
            if dur > 0:
                videos_with_duration.append((path, dur))
        
        if len(videos_with_duration) < 2:
            print("Not enough valid videos to compare")
            sys.exit(0)
        
        # Find duplicates
        duplicate_groups = find_duplicate_groups(videos_with_duration)
        
        # Save report
        report = {
            'directory': directory,
            'total_videos': len(videos_with_duration),
            'duplicate_sets': len(duplicate_groups),
            'sets': [
                {
                    'set_id': i + 1,
                    'videos': [
                        {
                            'path': v,
                            'filename': os.path.basename(v),
                            'duration': next((d for p, d in videos_with_duration if p == v), 0)
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
        organize_duplicates(directory, duplicate_groups, args.dry_run)
        
    finally:
        # Cleanup temp directory
        if TEMP_DIR and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


if __name__ == '__main__':
    main()
