import os
import subprocess
import json
from typing import Dict, Any, Tuple


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


def has_audio_stream(video_path: str) -> bool:
    """Check if video has an audio stream."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type',
              '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return 'audio' in result.stdout
    except Exception:
        return False


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Get video resolution as (width, height)."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
              '-show_entries', 'stream=width,height',
              '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        width = int(lines[0]) if lines and lines[0].isdigit() else 0
        height = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else 0
        return width, height
    except Exception:
        return 0, 0