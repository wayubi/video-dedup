import os
import json
import fcntl
import time
import hashlib
import tempfile
from typing import Dict, Optional
from dedup_config import (
    VideoFeatures, CACHE_DIR, CACHE_INDEX, CACHE_LOCK, CACHE_VERSION,
    SHORT_VIDEO_THRESHOLD, SHORT_SKIP_FIRST, SHORT_CLUSTER_SECONDS, AUDIO_CLUSTER_SECONDS,
    NUM_AUDIO_ANCHORS, SKIP_FIRST_SECONDS, AUDIO_SAMPLE_DURATION, TEMP_DIR
)


def get_cache_key(video_path: str) -> str:
    """Generate cache key for video path."""
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
    """Remove stale cache entries for videos that no longer exist or have changed."""
    index = load_cache_index()
    stale_keys = []
    for key, entry in index.items():
        video_path = entry.get("path", "")
        if not video_path:
            stale_keys.append(key)
            continue
        if not os.path.exists(video_path):
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
    index = load_cache_index()
    stat = os.stat(video_path)
    index[cache_key] = {
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
    save_cache_index(index)


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
    from dedup_ffmpeg import get_video_metadata
    from dedup_audio import extract_audio_sample, generate_audio_fingerprint
    from dedup_visual import extract_visual_samples_batch, generate_visual_fingerprint

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

                    audio_data = extract_audio_sample(video_path, timestamp, AUDIO_SAMPLE_DURATION)
                    if audio_data is not None:
                        fp = generate_audio_fingerprint(audio_data)
                        if len(fp) > 0:
                            cluster.append((timestamp, fp.tolist()))

                if cluster:
                    features["audio_fingerprints"].append(cluster)

    clusters = extract_visual_samples_batch(video_path, duration, effective_temp_dir)
    if clusters:
        features["visual_hashes"] = generate_visual_fingerprint(clusters)

    if features.get("duration", 0) > 0:
        save_features_to_cache(features)

    return features