import os
import subprocess
from typing import Optional
import numpy as np
import audioread
from lib.config import (
    TEMP_DIR, AUDIO_SAMPLE_DURATION, _FP_VALUES_PER_SECOND, MAX_AUDIO_OFFSET_SECONDS
)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_video_metadata(video_path: str):
    """Get all video metadata in a single ffprobe call."""
    import json
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


def extract_audio_sample(video_path: str, start_time: float, duration: float = AUDIO_SAMPLE_DURATION, temp_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Extract audio sample from video at a specific time, resampled to 16 kHz mono."""
    use_temp_dir = temp_dir if temp_dir is not None else TEMP_DIR
    if use_temp_dir is None:
        return None
    try:
        temp_wav = os.path.join(use_temp_dir, f"audio_{os.getpid()}_{os.path.basename(video_path)}_{start_time}.wav")
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


def compare_audio_clusters(cluster1, cluster2):
    """Find the best-matching fingerprint pair across two anchor clusters."""
    best_sim = 0.0
    best_i, best_j = 0, 0
    for i, (ts1, fp1_list) in enumerate(cluster1):
        for j, (ts2, fp2_list) in enumerate(cluster2):
            sim = compare_audio_fingerprints(np.array(fp1_list), np.array(fp2_list))
            if sim > best_sim:
                best_sim = sim
                best_i, best_j = i, j
    return best_sim, best_i, best_j


def compare_audio_fingerprints(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compare two audio fingerprints using sliding cross-correlation."""
    if len(fp1) == 0 or len(fp2) == 0:
        return 0.0

    std1, std2 = np.std(fp1), np.std(fp2)
    MIN_VAR = 0.1
    if std1 < MIN_VAR or std2 < MIN_VAR:
        return 0.0

    fp1 = (fp1 - np.mean(fp1)) / (std1 + 1e-10)
    fp2 = (fp2 - np.mean(fp2)) / (std2 + 1e-10)

    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]

    correlation = np.correlate(fp1, fp2, mode='full')

    max_shift = int(MAX_AUDIO_OFFSET_SECONDS * _FP_VALUES_PER_SECOND)
    center = len(correlation) // 2
    lo = max(0, center - max_shift)
    hi = min(len(correlation), center + max_shift + 1)
    best_raw = float(np.max(correlation[lo:hi]))

    similarity = best_raw / (min_len + 1e-10)
    return float(np.clip((similarity + 1) / 2, 0.0, 1.0))