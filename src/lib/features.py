import os
import json
import fcntl
import time
import hashlib
import tempfile
import subprocess
from typing import Dict, Optional, List
from pathlib import Path

from lib.config import (
    VideoFeatures, CACHE_DIR, CACHE_INDEX, CACHE_LOCK, CACHE_VERSION,
    SHORT_VIDEO_THRESHOLD, SHORT_SKIP_FIRST, SHORT_CLUSTER_SECONDS, AUDIO_CLUSTER_SECONDS,
    NUM_AUDIO_ANCHORS, SKIP_FIRST_SECONDS, AUDIO_SAMPLE_DURATION, TEMP_DIR,
    VIDEO_EXTENSIONS
)
from lib.audio import (
    get_video_duration, get_video_metadata, extract_audio_sample, generate_audio_fingerprint,
    has_audio_stream
)
from lib.visual import (
    get_video_resolution, extract_visual_samples_batch, generate_visual_fingerprint
)
import numpy as np
from scipy.ndimage import convolve


def find_videos(directory: str, include_subfolders: Optional[List[str]] = None, scan_mode: str = "RECURSIVE") -> List[str]:
    """Find all video files in directory based on scan_mode."""
    videos = []
    abs_directory = os.path.abspath(directory)
    deduped_folder = os.path.join(abs_directory, "__deduped")

    if scan_mode == "ROOT":
        for file in os.listdir(abs_directory):
            file_path = os.path.join(abs_directory, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(file_path)
    
    elif scan_mode == "TARGETED":
        if not include_subfolders:
            return []
        folders_to_scan = []
        for subfolder in include_subfolders:
            subfolder_path = (os.path.abspath(subfolder) if os.path.isabs(subfolder)
                              else os.path.abspath(os.path.join(abs_directory, subfolder)))
            if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
                continue
            if subfolder_path.startswith(deduped_folder):
                continue
            folders_to_scan.append(subfolder_path)
        
        for folder in folders_to_scan:
            for root, _, files in os.walk(folder):
                if root.startswith(deduped_folder):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(file_path)
    else:
        for root, _, files in os.walk(abs_directory):
            if root.startswith(deduped_folder):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(file_path)

    return videos


# Cache functions
def get_cache_key(video_path: str) -> str:
    return hashlib.md5(video_path.encode()).hexdigest()[:16]


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
    if entry.get("cache_version", 1) != CACHE_VERSION:
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


def prune_cache(directory: str) -> int:
    index = load_cache_index()
    stale_keys = []
    for key, entry in index.items():
        video_path = entry.get("path", "")
        if not video_path or not os.path.exists(video_path):
            stale_keys.append(key)
            continue
        try:
            stat = os.stat(video_path)
            if entry.get("size") != stat.st_size or entry.get("mtime") != stat.st_mtime:
                stale_keys.append(key)
        except Exception:
            stale_keys.append(key)
    for key in stale_keys:
        del index[key]
    if stale_keys:
        save_cache_index(index)
    return len(stale_keys)


def save_features_to_cache(features: VideoFeatures) -> None:
    video_path = features.get("path", "")
    if not video_path:
        return
    cache_key = get_cache_key(video_path)
    if not cache_key:
        return
    stat = os.stat(video_path)
    entry = {
        "path": os.path.abspath(video_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "cache_version": CACHE_VERSION,
        "duration": features.get("duration", 0.0),
        "resolution": list(features.get("resolution", (0, 0))),
        "has_audio": features.get("has_audio", False),
        "file_hash": features.get("file_hash", ""),
        "visual_hashes": features.get("visual_hashes", []),
        "audio_fingerprints": features.get("audio_fingerprints", []),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    for attempt in range(5):
        try:
            with open(CACHE_LOCK, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    index = {}
                    if os.path.exists(CACHE_INDEX):
                        try:
                            with open(CACHE_INDEX, 'r') as f:
                                index = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            pass
                    index[cache_key] = entry
                    with open(CACHE_INDEX, 'w') as f:
                        json.dump(index, f, indent=2)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return
        except (IOError, OSError):
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))


def compute_file_hash(video_path: str) -> str:
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
    import sys
    import traceback
    
    try:
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
            is_short_video = duration < SHORT_VIDEO_THRESHOLD
            effective_skip = SHORT_SKIP_FIRST if is_short_video else SKIP_FIRST_SECONDS
            cluster_seconds = SHORT_CLUSTER_SECONDS if is_short_video else AUDIO_CLUSTER_SECONDS
            num_anchors = NUM_AUDIO_ANCHORS

            available = duration - effective_skip

            if available > AUDIO_SAMPLE_DURATION:
                max_cluster = min(cluster_seconds, available / (num_anchors * 2))
                anchor_points = [(i + 0.5) / num_anchors * 0.9 + 0.05 for i in range(num_anchors)]

                for anchor_idx, anchor_point in enumerate(anchor_points):
                    anchor_timestamp = effective_skip + (available * anchor_point)
                    cluster = []

                    offsets = [0, -max_cluster, max_cluster]
                    for offset_idx, offset in enumerate(offsets):
                        timestamp = anchor_timestamp + offset
                        if timestamp < 0:
                            continue
                        if timestamp + AUDIO_SAMPLE_DURATION > duration:
                            continue
                        if abs(offset) > 0 and abs(offset) < 0.5:
                            continue

                        try:
                            audio_data = extract_audio_sample(video_path, timestamp, AUDIO_SAMPLE_DURATION, effective_temp_dir)
                            if audio_data is not None:
                                fp = generate_audio_fingerprint(audio_data)
                                if len(fp) > 0:
                                    cluster.append((timestamp, fp.tolist()))
                        except Exception as e:
                            print(f"[ERROR] Audio extraction failed at {timestamp}: {e}", file=sys.stderr)

                    if cluster:
                        features["audio_fingerprints"].append(cluster)

        try:
            clusters = extract_visual_samples_batch(video_path, duration, effective_temp_dir)
            if clusters:
                features["visual_hashes"] = generate_visual_fingerprint(clusters)
        except Exception as e:
            print(f"[ERROR] Visual extraction failed: {e}", file=sys.stderr)

        if features.get("duration", 0) > 0:
            save_features_to_cache(features)

        return features
    except Exception as e:
        print(f"[ERROR] extract_all_features failed for {video_path}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise


# Upscaling detection
def calculate_frequency_score(image):
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


def calculate_edge_sharpness_score(image):
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


def calculate_multiscale_score(video_path: str, width: int, height: int, temp_dir: Optional[str] = None) -> float:
    try:
        if temp_dir is None or max(width, height) < 1920:
            return 0.0
        duration = get_video_duration(video_path)
        timestamp = duration * 0.5
        temp_orig = os.path.join(temp_dir, f"upscale_orig_{os.path.basename(video_path)}.jpg")
        temp_down = os.path.join(temp_dir, f"upscale_down_{os.path.basename(video_path)}.jpg")

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

        from PIL import Image
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


def detect_upscaling(video_path: str, temp_dir: Optional[str] = None):
    from lib.config import MIN_UPSCALING_ANALYSIS_RESOLUTION, UPSCALING_CONFIDENCE_THRESHOLD
    width, height = get_video_resolution(video_path)
    if max(width, height) < MIN_UPSCALING_ANALYSIS_RESOLUTION:
        return False, 0.0, {"skipped": True, "resolution": (width, height)}
    try:
        if temp_dir is None:
            return False, 0.0, {"skipped": True}
        duration = get_video_duration(video_path)
        if duration <= 0:
            return False, 0.0, {"skipped": True}
        timestamp = max(SKIP_FIRST_SECONDS, duration * 0.3)
        temp_frame = os.path.join(temp_dir, f"upscaling_{os.path.basename(video_path)}.jpg")
        subprocess.run(['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                        '-vframes', '1', '-q:v', '2', temp_frame],
                       capture_output=True, timeout=15)
        if not os.path.exists(temp_frame):
            return False, 0.0, {"skipped": True}
        try:
            from PIL import Image
            img = Image.open(temp_frame).convert('RGB')
            arr = np.array(img).astype(np.float32)
            os.remove(temp_frame)
            freq_score = calculate_frequency_score(arr)
            edge_score = calculate_edge_sharpness_score(arr)
            multi_score = calculate_multiscale_score(video_path, width, height, temp_dir)
            combined = 0.35 * freq_score + 0.25 * edge_score + 0.40 * multi_score
            is_upscaled = combined > UPSCALING_CONFIDENCE_THRESHOLD
            scores = {"frequency": round(freq_score, 3), "edge_sharpness": round(edge_score, 3), "multiscale": round(multi_score, 3), "combined": round(combined, 3)}
            return bool(is_upscaled), float(combined), {"resolution": (width, height), "scores": scores}
        except Exception as e:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
            return False, 0.0, {"error": str(e)}
    except Exception as e:
        return False, 0.0, {"error": str(e)}