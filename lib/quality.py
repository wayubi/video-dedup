import os
from typing import Dict, Tuple
from lib.audio import get_video_metadata
from lib.features import compute_file_hash


def extract_video_metadata(video_path: str) -> Dict:
    """Extract full video metadata for quality scoring."""
    meta = {
        "path": video_path,
        "filename": os.path.basename(video_path),
        "file_size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        "modification_time": os.path.getmtime(video_path) if os.path.exists(video_path) else 0,
        "original_full_path": os.path.abspath(video_path),
        "file_hash": compute_file_hash(video_path),
    }
    
    ffprobe_meta = get_video_metadata(video_path)
    width, height = ffprobe_meta.get("resolution", (0, 0))
    meta["video"] = {
        "width": width,
        "height": height,
        "resolution": width * height,
        "duration": ffprobe_meta.get("duration", 0),
        "has_audio": ffprobe_meta.get("has_audio", False),
    }
    
    return meta


def calculate_quality_score(metadata: Dict) -> Tuple[int, str]:
    """Calculate quality score based on video properties."""
    video = metadata.get("video", {})
    width = video.get("width", 0)
    height = video.get("height", 0)
    resolution = width * height
    duration = video.get("duration", 0)
    
    score = 0
    reasons = []
    
    if resolution >= 3840 * 2160:
        score += 40
        reasons.append("4K")
    elif resolution >= 1920 * 1080:
        score += 30
        reasons.append("1080p")
    elif resolution >= 1280 * 720:
        score += 20
        reasons.append("720p")
    elif resolution >= 854 * 480:
        score += 10
        reasons.append("480p")
    
    if duration >= 600:
        score += 20
        reasons.append("10m+")
    elif duration >= 300:
        score += 15
        reasons.append("5m+")
    elif duration >= 60:
        score += 10
        reasons.append("1m+")
    
    if video.get("has_audio", False):
        score += 15
        reasons.append("audio")
    
    file_size = metadata.get("file_size_bytes", 0)
    if duration > 0:
        bitrate = (file_size * 8) / (duration * 1000)
        if bitrate >= 5000:
            score += 25
            reasons.append("high_bitrate")
        elif bitrate >= 2000:
            score += 15
            reasons.append("med_bitrate")
        elif bitrate >= 1000:
            score += 5
            reasons.append("low_bitrate")
    
    return score, "+".join(reasons) if reasons else "none"


def analyze_duplicate_set(videos_with_metadata) -> Dict[str, Dict]:
    """Analyze a duplicate set and determine KEEP/DELETE for each video."""
    results = {}
    if not videos_with_metadata:
        return results
    
    best_path = None
    best_score = -1
    best_width, best_height, best_bitrate = 0, 0, 0
    
    for video_path, metadata in videos_with_metadata:
        score, reason = calculate_quality_score(metadata)
        results[video_path] = {
            "quality_score": score,
            "reason": reason,
            "metadata": metadata,
        }
        
        video = metadata.get("video", {})
        width = video.get("width", 0)
        height = video.get("height", 0)
        
        file_size = metadata.get("file_size_bytes", 0)
        duration = video.get("duration", 0)
        bitrate = (file_size * 8) / (duration * 1000) if duration > 0 else 0
        
        if score > best_score:
            best_score = score
            best_path = video_path
            best_width, best_height = width, height
            best_bitrate = bitrate
    
    for video_path in results:
        result = results[video_path]
        if video_path == best_path:
            result["recommendation"] = "KEEP"
            result["reason"] = f"Best quality: {result.get('reason', '')}"
        else:
            score = result["quality_score"]
            vi = result["metadata"].get("video", {})
            v_width = vi.get("width", 0)
            v_height = vi.get("height", 0)
            
            file_size = result["metadata"].get("file_size_bytes", 0)
            duration = vi.get("duration", 0)
            v_bitrate = (file_size * 8) / (duration * 1000) if duration > 0 else 0

            reason = (f"Significantly lower quality (score {score} vs {best_score})"
                      if best_score - score >= 30 else
                      f"Lower quality (score {score} vs {best_score})")
            if v_width * v_height < best_width * best_height:
                reason += f"; lower resolution ({v_width}x{v_height} vs {best_width}x{best_height})"
            if v_bitrate < best_bitrate * 0.7 and v_bitrate > 0:
                reason += f"; lower bitrate ({int(v_bitrate)} vs {int(best_bitrate)} kbps)"

            result["recommendation"] = "DELETE"
            result["reason"] = reason
            result["better_alternative"] = {
                "filename": os.path.basename(best_path),
                "quality_score": best_score,
            }

    return results