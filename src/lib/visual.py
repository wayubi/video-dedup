import os
import subprocess
from typing import List, Tuple
from PIL import Image
import imagehash
from lib.config import NUM_VISUAL_ANCHORS, VISUAL_FRAME_THRESHOLD


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


def extract_visual_samples_batch(video_path: str, duration: float, temp_dir: str):
    """Extract frame clusters from video around anchor points."""
    if temp_dir is None or duration <= 0:
        return []

    from lib.config import SHORT_VIDEO_THRESHOLD, SHORT_SKIP_FIRST, SHORT_CLUSTER_SECONDS, VISUAL_CLUSTER_SECONDS, NUM_VISUAL_ANCHORS

    is_short_video = duration < SHORT_VIDEO_THRESHOLD
    effective_start = SHORT_SKIP_FIRST if is_short_video else 10
    cluster_seconds = SHORT_CLUSTER_SECONDS if is_short_video else VISUAL_CLUSTER_SECONDS
    num_anchors = NUM_VISUAL_ANCHORS

    available = duration - effective_start

    if available <= 0:
        return []

    max_cluster = min(cluster_seconds, available / (num_anchors * 2))

    clusters = []
    base_name = os.path.basename(video_path)

    anchor_points = [(i + 0.5) / num_anchors * 0.9 + 0.05 for i in range(num_anchors)]

    for anchor_idx, anchor_point in enumerate(anchor_points):
        anchor_timestamp = effective_start + (available * anchor_point)
        cluster = []

        offsets = [0, -max_cluster, max_cluster]

        for offset_idx, offset in enumerate(offsets):
            timestamp = anchor_timestamp + offset
            if timestamp < 0:
                continue
            if timestamp > duration:
                continue
            if abs(offset) > 0 and abs(offset) < 0.1:
                continue

            try:
                temp_frame = os.path.join(temp_dir, f"frame_{base_name}_{anchor_idx}_{offset_idx}.jpg")
                cmd = [
                    'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
                    '-vframes', '1', '-q:v', '2', temp_frame
                ]
                subprocess.run(cmd, capture_output=True, timeout=15)
                if os.path.exists(temp_frame):
                    img = Image.open(temp_frame)
                    cluster.append((timestamp, img))
                    os.remove(temp_frame)
            except Exception:
                pass

        if cluster:
            clusters.append(cluster)

    return clusters


def generate_visual_fingerprint(clusters: List[List[Tuple[float, Image.Image]]]) -> List[List[Tuple[float, List[str]]]]:
    """Generate region-based perceptual hashes per frame (3x3 grid)."""
    result: List[List[Tuple[float, List[str]]]] = []
    for cluster in clusters:
        cluster_hashes: List[Tuple[float, List[str]]] = []
        for timestamp, img in cluster:
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                w, h = img.size
                region_hashes = []
                for row in range(3):
                    y1 = int(h * row / 3)
                    y2 = int(h * (row + 1) / 3)
                    for col in range(3):
                        x1 = int(w * col / 3)
                        x2 = int(w * (col + 1) / 3)
                        region = img.crop((x1, y1, x2, y2))
                        region_hashes.append(str(imagehash.phash(region)))
                cluster_hashes.append((timestamp, region_hashes))
            except Exception:
                pass
        if cluster_hashes:
            result.append(cluster_hashes)
    return result


def hamming_distance(hash1: str, hash2: str) -> int:
    """Hamming distance between two hex hashes."""
    if len(hash1) != len(hash2):
        return 1000
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def compare_visual_fingerprints(hashes1: List[List[Tuple[float, List[str]]]], hashes2: List[List[Tuple[float, List[str]]]], verbose: bool = False) -> Tuple[float, List[str]]:
    """Compare visual fingerprints using cluster-based matching."""
    verbose_lines: List[str] = []

    if not hashes1 or not hashes2:
        return 0.0, verbose_lines

    n1, n2 = len(hashes1), len(hashes2)
    if n1 == 0 or n2 == 0:
        return 0.0, verbose_lines

    threshold_match = 15

    def frame_similarity(h1: List[str], h2: List[str]) -> float:
        if not h1 or not h2:
            return 0.0
        region_matches = 0
        for r1, r2 in zip(h1, h2):
            dist = hamming_distance(r1, r2)
            if dist <= threshold_match:
                region_matches += 1
        return region_matches / 9.0

    def compare_clusters(cluster1: List[Tuple[float, List[str]]], cluster2: List[Tuple[float, List[str]]]) -> Tuple[float, int, int]:
        best_sim = 0.0
        best_i, best_j = 0, 0
        for i, (ts1, h1) in enumerate(cluster1):
            for j, (ts2, h2) in enumerate(cluster2):
                sim = frame_similarity(h1, h2)
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j
        return best_sim, best_i, best_j

    best_count = 0
    num_anchors = min(n1, n2)

    for anchor_idx in range(num_anchors):
        cluster1 = hashes1[anchor_idx]
        anchor_ts = cluster1[0][0] if cluster1 else 0

        best_cluster_sim = 0.0
        best_cluster_idx = anchor_idx
        best_i_in_cluster, best_j_in_cluster = 0, 0

        lo = max(0, anchor_idx - 3)
        hi = min(n2, anchor_idx + 4)

        for other_anchor_idx in range(lo, hi):
            cluster2 = hashes2[other_anchor_idx]
            sim, i_in_cluster, j_in_cluster = compare_clusters(cluster1, cluster2)
            if sim > best_cluster_sim:
                best_cluster_sim = sim
                best_cluster_idx = other_anchor_idx
                best_i_in_cluster, best_j_in_cluster = i_in_cluster, j_in_cluster

        if verbose:
            verbose_lines.append(f"      Frame {anchor_idx}: anchor={anchor_ts:.1f}s")
            for i, (ts1, h1) in enumerate(cluster1):
                for j, (ts2, h2) in enumerate(hashes2[best_cluster_idx]):
                    sim = frame_similarity(h1, h2)
                    verbose_lines.append(f"        Frame {i}a: ts={ts1:.1f}s vs Frame {j}a: ts={ts2:.1f}s: similarity={sim:.4f}")
            best_ts = hashes2[best_cluster_idx][best_j_in_cluster][0] if hashes2[best_cluster_idx] else 0
            result = "PASS" if best_cluster_sim > VISUAL_FRAME_THRESHOLD else "FAIL"
            verbose_lines.append(f"        Best for frame {anchor_idx}: frame {best_j_in_cluster}a ts={best_ts:.1f}s similarity={best_cluster_sim:.4f} (threshold={VISUAL_FRAME_THRESHOLD:.4f}, result={result})")

        if best_cluster_sim > VISUAL_FRAME_THRESHOLD:
            best_count += 1

    return best_count / max(n1, n2), verbose_lines