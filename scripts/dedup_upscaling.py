import os
import subprocess
from typing import Tuple, Dict
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from dedup_config import TEMP_DIR, MIN_UPSCALING_ANALYSIS_RESOLUTION
from dedup_ffmpeg import get_video_duration, get_video_resolution


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
            return False, 0.0, {"skipped": True, "reason": "no_temp_dir"}

        duration = get_video_duration(video_path)
        if duration <= 0:
            return False, 0.0, {"skipped": True, "reason": "no_duration"}

        timestamp = duration * 0.1
        temp_frame = os.path.join(TEMP_DIR, f"upscaling_{os.path.basename(video_path)}.jpg")
        subprocess.run(['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                        '-vframes', '1', '-q:v', '2', temp_frame],
                       capture_output=True, timeout=15)

        if not os.path.exists(temp_frame):
            return False, 0.0, {"skipped": True, "reason": "frame_extraction_failed"}

        try:
            img = Image.open(temp_frame).convert('RGB')
            arr = np.array(img).astype(np.float32)
            os.remove(temp_frame)

            freq_score = calculate_frequency_score(arr)
            edge_score = calculate_edge_sharpness_score(arr)
            multi_score = calculate_multiscale_score(video_path, width, height)

            scores = {"frequency": freq_score, "edge": edge_score, "multiscale": multi_score}
            avg_score = (freq_score + edge_score + multi_score) / 3

            confidence = avg_score
            is_upscaled = confidence >= 0.65

            return is_upscaled, confidence, {"resolution": (width, height), "scores": scores}
        except Exception as e:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
            return False, 0.0, {"error": str(e)}
    except Exception as e:
        return False, 0.0, {"error": str(e)}