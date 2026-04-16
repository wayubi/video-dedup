#!/usr/bin/env python3
"""Video Duplicate Detector - Main entrypoint."""
import os
import sys
import json
import shutil
import tempfile
import argparse
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from lib.config import (
    VideoFeatures, TEMP_DIR, LENGTH_TOLERANCE, AUDIO_THRESHOLD, AUDIO_REQUIRED_MATCHES,
    VISUAL_FRAME_THRESHOLD
)
from lib.utils import (
    format_bytes, scan_deduped_folders, get_associated_video_path,
    find_delete_files, find_all_files, delete_file_pair, restore_file, cleanup_empty_deduped_folders
)
from lib.features import find_videos, prune_cache, extract_all_features
from lib.audio import compare_audio_fingerprints
from lib.visual import generate_visual_fingerprint, compare_visual_fingerprints, extract_visual_samples_batch
from lib.quality import extract_video_metadata, calculate_quality_score, analyze_duplicate_set
from lib.features import detect_upscaling
from lib.config import UPSCALING_CONFIDENCE_THRESHOLD, MIN_UPSCALING_ANALYSIS_RESOLUTION

TEMP_DIR = None


def compare_features(f1: VideoFeatures, f2: VideoFeatures, verbose: bool = False):
    v1_name = os.path.basename(f1.get("path", ""))
    v2_name = os.path.basename(f2.get("path", ""))
    dur1, dur2 = f1.get("duration", 0), f2.get("duration", 0)
    verbose_lines = []

    if verbose:
        verbose_lines.append(f"  Comparing {v1_name} vs {v2_name}")

    if dur1 <= 0 or dur2 <= 0:
        return False, "no_duration", verbose_lines

    hash1, hash2 = f1.get("file_hash", ""), f2.get("file_hash", "")
    if hash1 and hash2 and hash1 == hash2:
        return True, "file_hash", verbose_lines

    duration_diff = abs(dur1 - dur2) / max(dur1, dur2, 1)
    if duration_diff > LENGTH_TOLERANCE:
        return False, "duration_mismatch", verbose_lines

    if f1.get("has_audio") and f2.get("has_audio"):
        fps1, fps2 = f1.get("audio_fingerprints", []), f2.get("audio_fingerprints", [])
        if fps1 and fps2:
            matches = 0
            for i in range(min(len(fps1), len(fps2))):
                best_sim = 0.0
                for j in range(len(fps2[i])):
                    ts2, fp2_list = fps2[i][j]
                    for k in range(len(fps1[i])):
                        ts1, fp1_list = fps1[i][k]
                        sim = compare_audio_fingerprints(np.array(fp1_list), np.array(fp2_list))
                        if sim > best_sim:
                            best_sim = sim
                if best_sim >= AUDIO_THRESHOLD:
                    matches += 1
            if matches >= AUDIO_REQUIRED_MATCHES:
                return True, "audio_fingerprint", verbose_lines

    clusters1 = extract_visual_samples_batch(f1.get("path", ""), dur1, TEMP_DIR)
    clusters2 = extract_visual_samples_batch(f2.get("path", ""), dur2, TEMP_DIR)
    if clusters1 and clusters2:
        hashes1 = generate_visual_fingerprint(clusters1)
        hashes2 = generate_visual_fingerprint(clusters2)
        sim, _ = compare_visual_fingerprints(hashes1, hashes2)
        if sim >= VISUAL_FRAME_THRESHOLD:
            return True, "visual_fingerprint", verbose_lines

    return False, "no_match", verbose_lines


def find_duplicate_groups_with_features(features_list, verbose=False):
    n = len(features_list)
    if n < 2:
        return []

    groups = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        group = [features_list[i]["path"]]
        for j in range(i + 1, n):
            if j in used:
                continue
            is_dup, _, _ = compare_features(features_list[i], features_list[j], verbose)
            if is_dup:
                group.append(features_list[j]["path"])
                used.add(j)

        if len(group) > 1:
            groups.append(group)
            used.add(i)

    return groups


def organize_duplicates(directory, duplicate_groups, dry_run=False, create_markers=False):
    if not duplicate_groups:
        print("No duplicates found!")
        return

    print(f"\nFound {len(duplicate_groups)} duplicate sets")
    deduped_base = os.path.join(directory, "__deduped")

    for i, group in enumerate(duplicate_groups, 1):
        folder_name = f"duplicate_set_{i:03d}"
        folder_path = os.path.join(deduped_base, folder_name)
        print(f"\n{folder_name}: {len(group)} videos")

        videos_with_metadata = [(v, extract_video_metadata(v)) for v in group]
        analysis_results = analyze_duplicate_set(videos_with_metadata)

        for video_path in group:
            rel_path = os.path.relpath(video_path, directory)
            analysis = analysis_results.get(video_path, {})
            print(f"  - {rel_path} [{analysis.get('recommendation', 'UNKNOWN')}, "
                  f"score: {analysis.get('quality_score', 0)}]")

        if not dry_run:
            os.makedirs(folder_path, exist_ok=True)
            for video_path in group:
                try:
                    filename = os.path.basename(video_path)
                    dest_path = os.path.join(folder_path, filename)
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(folder_path, f"{base_name}_{counter:02d}{ext}")
                        counter += 1

                    analysis = analysis_results.get(video_path, {})
                    metadata = analysis.get("metadata", {})
                    json_data = {
                        "recommendation": analysis.get("recommendation", "UNKNOWN"),
                        "original_full_path": metadata.get("original_full_path", os.path.abspath(video_path)),
                        "quality_score": analysis.get("quality_score", 0),
                        "reason": analysis.get("reason", ""),
                        "filename": filename,
                        "file_size_bytes": metadata.get("file_size_bytes", 0),
                        "modification_time": metadata.get("modification_time", 0),
                        **metadata,
                    }

                    json_filename = f"{os.path.basename(dest_path)}.json"
                    with open(os.path.join(folder_path, json_filename), 'w') as f:
                        json.dump(json_data, f, indent=2)

                    if create_markers:
                        rec = analysis.get("recommendation", "")
                        ext_marker = ".keep" if rec == "KEEP" else ".delete" if rec == "DELETE" else None
                        if ext_marker:
                            open(os.path.join(folder_path, f"{os.path.basename(dest_path)}{ext_marker}"), 'w').close()

                    shutil.move(video_path, dest_path)
                    print(f"  -> Moved to __deduped/{folder_name}")
                except Exception as e:
                    print(f"  ERROR moving {video_path}: {e}")


def run_scan(args):
    global TEMP_DIR
    TEMP_DIR = tempfile.mkdtemp(prefix="video_dedup_")

    try:
        base_dir = os.path.abspath(args.directory)
        scan_mode = "RECURSIVE"
        include_folders = None

        if args.no_subfolders:
            scan_mode = "ROOT"
        elif args.include_folders:
            scan_mode = "TARGETED"
            include_folders = args.include_folders

        if scan_mode == "ROOT":
            print(f"Scanning {base_dir} (root directory only)...")
        elif scan_mode == "TARGETED":
            print(f"Scanning {base_dir} with specific subfolders:")
            for p in include_folders:
                print(f"  - {p}")
        else:
            print(f"Scanning {base_dir} and all subfolders...")

        video_paths = find_videos(base_dir, include_folders, scan_mode=scan_mode)
        if not video_paths:
            print("No videos found!")
            return

        video_paths = sorted(video_paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
        print(f"Found {len(video_paths)} videos")

        upscaling_results = {}
        if args.detect_upscaling:
            from lib.config import UPSCALING_CONFIDENCE_THRESHOLD
            print("\n" + "=" * 60)
            print("UPSCALING DETECTION MODE")
            print("=" * 60)
            total = len(video_paths)
            for idx, path in enumerate(video_paths):
                is_upscaled, confidence, details = detect_upscaling(path)
                upscaling_results[path] = {
                    "is_upscaled": is_upscaled,
                    "confidence": round(confidence, 3),
                    "details": details,
                }
                if (idx + 1) % 10 == 0 or idx + 1 == total:
                    pct = int(100 * (idx + 1) / total)
                    print(f"Upscaling analysis: {idx + 1}/{total} ({pct}%)", flush=True)
            upscaled_count = sum(1 for r in upscaling_results.values() if r["is_upscaled"])
            print(f"\nUpscaling: {upscaled_count} of {len(upscaling_results)} flagged")
            for path, result in upscaling_results.items():
                if result["is_upscaled"]:
                    w, h = result["details"].get("resolution", (0, 0))
                    print(f"  - {os.path.basename(path)} ({w}x{h}) [conf: {result['confidence']:.2f}]")

        if args.wipe_cache:
            from lib.config import CACHE_DIR, CACHE_LOCK, CACHE_INDEX
            try:
                if os.path.exists(CACHE_LOCK):
                    os.remove(CACHE_LOCK)
                if os.path.exists(CACHE_INDEX):
                    os.remove(CACHE_INDEX)
                print(f"Deleted cache contents: {CACHE_DIR}")
            except Exception as e:
                print(f"Warning: Could not fully delete cache: {e}")

        if args.prune_cache:
            pruned = prune_cache(base_dir)
            print(f"{'Pruned ' if pruned else ''}{pruned} stale cache entries")

        n_workers = os.cpu_count() or 4
        print(f"Using {n_workers} workers")

        features_list = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_all_features, path): path for path in video_paths}
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                try:
                    features = future.result()
                    if features.get("duration", 0) > 0:
                        features_list.append(features)
                except Exception:
                    pass
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"Extracting features: {completed}/{total}", flush=True)

        if len(features_list) < 2:
            print("Not enough valid videos to compare")
            return

        duplicate_groups = find_duplicate_groups_with_features(features_list, verbose=args.verbose)

        scan_report = {
            'scan_mode': scan_mode,
            'directory': base_dir,
            'total_videos_analyzed': len(features_list),
            'duplicate_sets_found': len(duplicate_groups),
            'sets': [{'set_id': i + 1, 'videos': [{'path': v} for v in group]}
                    for i, group in enumerate(duplicate_groups)],
        }

        if args.detect_upscaling:
            from lib.config import UPSCALING_CONFIDENCE_THRESHOLD, MIN_UPSCALING_ANALYSIS_RESOLUTION
            upscaled_count = sum(1 for r in upscaling_results.values() if r["is_upscaled"])
            scan_report['upscaling_analysis'] = {
                'enabled': True,
                'threshold': UPSCALING_CONFIDENCE_THRESHOLD,
                'min_resolution_analyzed': MIN_UPSCALING_ANALYSIS_RESOLUTION,
                'summary': {
                    'total_analyzed': len(upscaling_results),
                    'upscaled_detected': upscaled_count,
                    'analysis_methods': ['frequency_analysis', 'edge_sharpness', 'multiscale_comparison'],
                },
            }

        report_path = os.path.join(base_dir, args.report)
        with open(report_path, 'w') as f:
            json.dump(scan_report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        organize_duplicates(base_dir, duplicate_groups, args.dry_run, args.create_markers)

    finally:
        if TEMP_DIR and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


def run_delete(args):
    base_dir = os.path.abspath(args.directory)
    sets = scan_deduped_folders(base_dir)

    if not sets:
        print(f"No __deduped folder found in {base_dir}")
        return

    all_candidates = []
    for set_path in sets:
        candidates = find_delete_files(set_path)
        all_candidates.extend([(set_path, v, j, m) for v, j, m, mdata in candidates])

    if not all_candidates:
        print("No DELETE files found.")
        return

    total_size = sum(os.path.getsize(v) for _, v, _, _ in all_candidates)
    mode = "DRY RUN" if not args.confirm else "LIVE DELETE"
    print(f"\n{'='*60}")
    print(f"DEDUP DELETE - {mode}")
    print(f"{'='*60}")
    print(f"\nFound {len(all_candidates)} files to delete")
    print(f"Total space: {format_bytes(total_size)}")

    for set_path, video_path, json_path, metadata in all_candidates:
        if args.confirm:
            success, _ = delete_file_pair(video_path, json_path)
            if success:
                print(f"  Deleted: {os.path.basename(video_path)}")
        else:
            print(f"  Would delete: {os.path.basename(video_path)}")

    if not args.confirm:
        print("\nDRY RUN - no files deleted. Use --confirm to delete.")

    removed, _ = cleanup_empty_deduped_folders(base_dir)
    print(f"Cleaned up {len(removed)} folders")


def run_restore(args):
    base_dir = os.path.abspath(args.directory)
    sets = scan_deduped_folders(base_dir)

    if not sets:
        print(f"No __deduped folder found in {base_dir}")
        return

    all_files = []
    for set_path in sets:
        files = find_all_files(set_path)
        all_files.extend([(set_path, v, j, m, md) for v, j, m, md in files])

    if not all_files:
        print("No files to restore.")
        return

    total_size = sum(os.path.getsize(v) for _, v, _, _, _ in all_files)
    print(f"\n{'='*60}")
    print(f"DEDUP RESTORE")
    print(f"{'='*60}")
    print(f"\nFound {len(all_files)} files to restore")
    print(f"Total size: {format_bytes(total_size)}")

    restored = []
    for set_path, video_path, json_path, marker_path, metadata in all_files:
        success, dest, error, collision = restore_file(video_path, json_path, marker_path, metadata)
        if success:
            name = os.path.basename(video_path)
            print(f"  Restored: {name}" + (" (collision)" if collision else ""))
            restored.append(video_path)
        else:
            print(f"  Failed: {os.path.basename(video_path)} - {error}")

    removed, _ = cleanup_empty_deduped_folders(base_dir)
    print(f"\nRestored {len(restored)} files, cleaned up {len(removed)} folders")


def main():
    parser = argparse.ArgumentParser(description='Video Duplicate Detector')
    parser.add_argument('directory', help='Directory containing videos')
    
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--scan', action='store_true', help='Scan for duplicates (default)')
    mode_group.add_argument('--delete', action='store_true', help='Delete candidates')
    mode_group.add_argument('--restore', action='store_true', help='Restore files')
    
    parser.add_argument('--dry-run', action='store_true', help='Show without moving')
    parser.add_argument('--report', type=str, default='duplicates_report.json', help='Report path')
    parser.add_argument('--detect-upscaling', action='store_true', help='Detect upscaling')
    parser.add_argument('--create-markers', action='store_true', help='Create markers')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--no-subfolders', action='store_true', help='Root only')
    group.add_argument('--include-folders', nargs='+', help='Specific folders')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    parser.add_argument('--prune-cache', action='store_true', help='Prune cache')
    parser.add_argument('--wipe-cache', action='store_true', help='Wipe cache')
    parser.add_argument('--confirm', action='store_true', help='Confirm delete')
    
    args = parser.parse_args()
    
    if not (args.delete or args.restore):
        args.scan = True
    
    if args.scan:
        run_scan(args)
    elif args.delete:
        run_delete(args)
    elif args.restore:
        run_restore(args)


if __name__ == '__main__':
    main()