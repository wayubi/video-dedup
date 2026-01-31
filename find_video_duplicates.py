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
    if np.std(fp1) == 0 or np.std(fp2) == 0:
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
                # Resolve relative path
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
                if similarity > 0.75:  # Threshold for match
                    samples_match += 1
        
        # Consider it a match if at least half the samples match
        if total_samples > 0 and samples_match >= total_samples / 2:
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


def organize_duplicates(directory: str, duplicate_groups: List[List[str]], dry_run: bool = False):
    """Move duplicate videos into .deduped/numbered folders."""
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
        for video in group:
            rel_path = os.path.relpath(video, directory)
            print(f"  - {rel_path}")
        
        if not dry_run:
            # Create .deduped folder and set folder
            os.makedirs(folder_path, exist_ok=True)
            
            # Move all videos in the group
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
                    
                    shutil.move(video_path, dest_path)
                    print(f"  -> Moved to .deduped/{folder_name}")
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
