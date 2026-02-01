#!/usr/bin/env python3
"""
Dedup Delete Script
Deletes all videos marked as DELETE_CANDIDATE from the .deduped folder.
Dry-run by default. Use --confirm to actually delete files.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def format_bytes(bytes_val: float) -> str:
    """Convert bytes to human-readable format."""
    val = float(bytes_val)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if val < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PB"


def scan_deduped_folders(base_dir: str) -> List[str]:
    """Scan .deduped folder for all duplicate set folders."""
    deduped_path = os.path.join(base_dir, ".deduped")
    
    if not os.path.exists(deduped_path):
        return []
    
    sets = []
    for item in os.listdir(deduped_path):
        item_path = os.path.join(deduped_path, item)
        if os.path.isdir(item_path) and item.startswith("duplicate_set_"):
            sets.append(item_path)
    
    return sorted(sets)


def load_video_metadata(json_path: str) -> Optional[Dict]:
    """Load and validate video metadata JSON."""
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate required fields
        if 'recommendation' not in metadata:
            return None
        
        return metadata
    except Exception as e:
        return None


def get_associated_video_path(json_path: str) -> Optional[str]:
    """Get the path to the video file associated with a JSON file."""
    # Video file is the JSON filename without the .json extension
    if json_path.endswith('.json'):
        video_path = json_path[:-5]
        if os.path.exists(video_path):
            return video_path
    return None


def find_delete_candidates(set_path: str) -> List[Tuple[str, str, str, Dict]]:
    """
    Find all DELETE_CANDIDATE files in a duplicate set.
    Returns list of (video_path, json_path, marker_path, metadata) tuples.
    """
    candidates = []
    
    for file in os.listdir(set_path):
        if file.endswith('.json'):
            json_path = os.path.join(set_path, file)
            metadata = load_video_metadata(json_path)
            
            if metadata and metadata.get('recommendation') == 'DELETE_CANDIDATE':
                video_path = get_associated_video_path(json_path)
                if video_path:
                    # Find the marker file (.delete)
                    marker_path = video_path + '.delete'
                    if not os.path.exists(marker_path):
                        marker_path = None
                    candidates.append((video_path, json_path, marker_path, metadata))
    
    return candidates


def delete_file_pair(video_path: str, json_path: str, marker_path: Optional[str] = None) -> Tuple[bool, str]:
    """Delete a video file, its associated JSON, and marker file. Returns (success, error_message)."""
    try:
        # Delete video file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Delete JSON file
        if os.path.exists(json_path):
            os.remove(json_path)
        
        # Delete marker file if it exists
        if marker_path and os.path.exists(marker_path):
            os.remove(marker_path)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Delete videos marked as DELETE_CANDIDATE from .deduped folder'
    )
    parser.add_argument('directory', help='Base directory containing .deduped folder')
    parser.add_argument('--confirm', action='store_true',
                        help='Actually delete files (dry-run by default)')
    parser.add_argument('--report', type=str, default='dedup_delete_report.json',
                        help='Path to save deletion report')
    
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.directory)
    
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory")
        sys.exit(1)
    
    # Scan for duplicate sets
    sets = scan_deduped_folders(base_dir)
    
    if not sets:
        print(f"No .deduped folder found in {base_dir}")
        print("Nothing to delete.")
        sys.exit(0)
    
    # Find all delete candidates
    all_candidates = []
    for set_path in sets:
        candidates = find_delete_candidates(set_path)
        all_candidates.extend([(set_path, video, json_file, marker_file, metadata) 
                               for video, json_file, marker_file, metadata in candidates])
    
    if not all_candidates:
        print("No DELETE_CANDIDATE files found.")
        print("All videos are marked as KEEP.")
        sys.exit(0)
    
    # Calculate totals
    total_size = sum(os.path.getsize(video) for _, video, _, _, _ in all_candidates)
    
    # Show summary
    mode = "DRY RUN" if not args.confirm else "LIVE DELETE"
    print(f"\n{'='*60}")
    print(f"DEDUP DELETE - {mode}")
    print(f"{'='*60}")
    print(f"\nFound {len(all_candidates)} files to delete across {len(sets)} sets")
    print(f"Total space to be freed: {format_bytes(total_size)}")
    print(f"\nFiles to delete:")
    
    for set_path, video_path, json_path, marker_path, metadata in all_candidates:
        set_name = os.path.basename(set_path)
        video_name = os.path.basename(video_path)
        size = os.path.getsize(video_path)
        reason = metadata.get('reason', 'No reason specified')
        
        print(f"\n  [{set_name}] {video_name}")
        print(f"    Size: {format_bytes(size)}")
        print(f"    Reason: {reason}")
        print(f"    Original location: {metadata.get('original_full_path', 'Unknown')}")
    
    print(f"\n{'='*60}")
    
    if not args.confirm:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("Use --confirm to actually delete these files.")
        print(f"\nReport will be saved to: {args.report}")
    else:
        print("\nDeleting files...")
    
    # Process deletions
    deleted_files = []
    failed_files = []
    
    for set_path, video_path, json_path, marker_path, metadata in all_candidates:
        set_name = os.path.basename(set_path)
        video_name = os.path.basename(video_path)
        size = os.path.getsize(video_path)
        
        if args.confirm:
            success, error = delete_file_pair(video_path, json_path, marker_path)
        else:
            success, error = True, ""  # Simulate success in dry-run
        
        file_info = {
            "set": set_name,
            "video": os.path.relpath(video_path, base_dir),
            "json": os.path.relpath(json_path, base_dir),
            "size_bytes": size,
            "original_path": metadata.get('original_full_path', 'Unknown'),
            "reason": metadata.get('reason', 'No reason specified')
        }
        
        if success:
            deleted_files.append(file_info)
            if args.confirm:
                print(f"  ✓ Deleted: {video_name}")
        else:
            file_info["error"] = error
            failed_files.append(file_info)
            if args.confirm:
                print(f"  ✗ Failed: {video_name} - {error}")
    
    # Clean up empty duplicate set folders (only in confirm mode)
    if args.confirm:
        for set_path in sets:
            remaining_files = [f for f in os.listdir(set_path) 
                             if os.path.isfile(os.path.join(set_path, f))]
            if not remaining_files:
                try:
                    os.rmdir(set_path)
                    print(f"  ✓ Removed empty folder: {os.path.basename(set_path)}")
                except Exception as e:
                    print(f"  ✗ Could not remove folder: {os.path.basename(set_path)} - {e}")
    
    # Calculate actual freed space
    freed_space = sum(f["size_bytes"] for f in deleted_files)
    
    # Generate report
    report = {
        "action": "delete_candidates",
        "timestamp": datetime.now().isoformat() + "Z",
        "base_directory": base_dir,
        "dry_run": not args.confirm,
        "summary": {
            "sets_processed": len(sets),
            "files_deleted": len(deleted_files),
            "files_failed": len(failed_files),
            "space_freed_bytes": freed_space,
            "space_freed_human": format_bytes(freed_space)
        },
        "deleted_files": deleted_files,
        "failed_files": failed_files
    }
    
    report_path = os.path.join(base_dir, args.report)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files deleted: {len(deleted_files)}")
    print(f"Files failed: {len(failed_files)}")
    print(f"Space freed: {format_bytes(freed_space)}")
    print(f"\nReport saved to: {report_path}")
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files could not be deleted. Check report for details.")
        sys.exit(1)
    elif args.confirm:
        print(f"\n✓ All DELETE_CANDIDATE files have been removed.")


if __name__ == '__main__':
    main()
