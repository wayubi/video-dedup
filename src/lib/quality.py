import os
import json
import subprocess
from typing import Any, Dict, List, Tuple


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


def calculate_quality_score(metadata: Dict) -> Tuple[int, str]:
    score = 0
    reasons = []

    video_info = metadata.get("video", {})
    if not video_info:
        return 0, "No video stream found"

    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    resolution_pixels = width * height

    score += int(40 * resolution_pixels / (1920 * 1080))
    if resolution_pixels >= 1920 * 1080:
        score += 25
        reasons.append(f"1080p+ ({width}x{height})")
    elif resolution_pixels >= 1280 * 720:
        score += 15
        reasons.append(f"720p ({width}x{height})")
    else:
        reasons.append(f"sub-720p ({width}x{height})")

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

    hdr = video_info.get("hdr_format")
    if hdr:
        score += 15; reasons.append(f"HDR ({hdr})")

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
            "recommendation": "DELETE", "reason": "", "better_alternative": None,
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

        result["recommendation"] = "DELETE"
        result["reason"] = reason
        result["better_alternative"] = {
            "filename": os.path.basename(best_path),
            "quality_score": best_score,
            "reason": f"Better quality: {best_res}, {best_codec}, {best_fps}fps, score {best_score}",
        }

    return results
