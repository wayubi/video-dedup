#!/usr/bin/env python3
"""
Dedup Restore Script
Restores all videos marked as KEEP from .deduped folder back to their original locations.
After successful restore, removes files from .deduped and cleans up empty folders.
"""

import os
import sys
import json
import shutil
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
        if 'original_full_path' not in metadata:
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


def find_all_files(set_path: str) -> List[Tuple[str, str, Optional[str], Dict]]:
    """
    Find all files in a duplicate set (not just KEEP files).
    Returns list of (video_path, json_path, marker_path, metadata) tuples.
    """
    all_files = []
    
    for file in os.listdir(set_path):
        if file.endswith('.json'):
            json_path = os.path.join(set_path, file)
            metadata = load_video_metadata(json_path)
            
            if metadata:  # Restore ALL files with valid metadata, not just KEEP
                video_path = get_associated_video_path(json_path)
                if video_path:
                    # Find the marker file (.keep or .delete)
                    marker_path = None
                    keep_marker = video_path + '.keep'
                    delete_marker = video_path + '.delete'
                    if os.path.exists(keep_marker):
                        marker_path = keep_marker
                    elif os.path.exists(delete_marker):
                        marker_path = delete_marker
                    all_files.append((video_path, json_path, marker_path, metadata))
    
    return all_files


def restore_file(video_path: str, json_path: str, marker_path: Optional[str], metadata: Dict) -> Tuple[bool, str, str, bool]:
    """
    Restore a video file to its original location.
    Returns (success, destination_path, error_message, collision_handled).
    """
    try:
        original_path = metadata.get('original_full_path')
        if not original_path:
            return False, "", "No original path in metadata", False
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(original_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Handle filename collision
        destination = original_path
        collision_handled = False
        
        if os.path.exists(destination):
            # File already exists at destination, create unique name
            base_name, ext = os.path.splitext(original_path)
            counter = 1
            while os.path.exists(destination):
                destination = f"{base_name}_restored_{counter:03d}{ext}"
                counter += 1
            collision_handled = True
        
        # Move the file
        shutil.move(video_path, destination)
        
        # Delete the JSON file
        if os.path.exists(json_path):
            os.remove(json_path)
        
        # Delete the marker file if it exists
        if marker_path and os.path.exists(marker_path):
            os.remove(marker_path)
        
        return True, destination, "", collision_handled
        
    except Exception as e:
        return False, "", str(e), False


def main():
    parser = argparse.ArgumentParser(
        description='Restore videos marked as KEEP from .deduped folder to original locations'
    )
    parser.add_argument('directory', help='Base directory containing .deduped folder')
    parser.add_argument('--report', type=str, default='dedup_restore_report.json',
                        help='Path to save restoration report')
    
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.directory)
    
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory")
        sys.exit(1)
    
    # Scan for duplicate sets
    sets = scan_deduped_folders(base_dir)
    
    if not sets:
        print(f"No .deduped folder found in {base_dir}")
        print("Nothing to restore.")
        sys.exit(0)
    
    # Find all files in duplicate sets
    all_files = []
    for set_path in sets:
        files = find_all_files(set_path)
        all_files.extend([(set_path, video, json_file, marker_file, metadata) 
                          for video, json_file, marker_file, metadata in files])
    
    if not all_files:
        print("No files found in .deduped folder.")
        print("Nothing to restore.")
        sys.exit(0)
    
    # Calculate totals
    total_size = sum(os.path.getsize(video) for _, video, _, _, _ in all_files)
    
    # Show summary
    print(f"\n{'='*60}")
    print(f"DEDUP RESTORE")
    print(f"{'='*60}")
    print(f"\nFound {len(all_files)} files to restore across {len(sets)} sets")
    print(f"Total size: {format_bytes(total_size)}")
    print(f"\nFiles to restore:")
    
    for set_path, video_path, json_path, marker_path, metadata in all_files:
        set_name = os.path.basename(set_path)
        video_name = os.path.basename(video_path)
        size = os.path.getsize(video_path)
        original_path = metadata.get('original_full_path', 'Unknown')
        quality_score = metadata.get('quality_score', 0)
        recommendation = metadata.get('recommendation', 'Unknown')
        
        print(f"\n  [{set_name}] {video_name}")
        print(f"    Size: {format_bytes(size)}")
        print(f"    Quality Score: {quality_score}")
        print(f"    Recommendation: {recommendation}")
        print(f"    Restore to: {original_path}")
    
    print(f"\n{'='*60}")
    print("\nRestoring files...")
    
    # Process restorations
    restored_files = []
    failed_files = []
    
    for set_path, video_path, json_path, marker_path, metadata in all_files:
        set_name = os.path.basename(set_path)
        video_name = os.path.basename(video_path)
        size = os.path.getsize(video_path)
        
        success, destination, error, collision = restore_file(video_path, json_path, marker_path, metadata)
        
        file_info = {
            "set": set_name,
            "source": os.path.relpath(video_path, base_dir),
            "destination": destination,
            "size_bytes": size,
            "original_path": metadata.get('original_full_path', 'Unknown'),
            "quality_score": metadata.get('quality_score', 0),
            "collision_handled": collision
        }
        
        if success:
            restored_files.append(file_info)
            if collision:
                print(f"  ✓ Restored: {video_name} (collision handled)")
            else:
                print(f"  ✓ Restored: {video_name}")
        else:
            file_info["error"] = error
            failed_files.append(file_info)
            print(f"  ✗ Failed: {video_name} - {error}")
    
    # Clean up empty duplicate set folders
    print("\nCleaning up empty folders...")
    for set_path in sets:
        try:
            remaining_files = [f for f in os.listdir(set_path) 
                             if os.path.isfile(os.path.join(set_path, f))]
            if not remaining_files:
                os.rmdir(set_path)
                print(f"  ✓ Removed empty folder: {os.path.basename(set_path)}")
        except Exception as e:
            print(f"  ✗ Could not remove folder: {os.path.basename(set_path)} - {e}")
    
    # Remove .deduped folder if empty
    deduped_path = os.path.join(base_dir, ".deduped")
    try:
        if os.path.exists(deduped_path):
            remaining = os.listdir(deduped_path)
            if not remaining:
                os.rmdir(deduped_path)
                print(f"  ✓ Removed empty .deduped folder")
            else:
                print(f"  ℹ️  .deduped folder still contains {len(remaining)} items")
    except Exception as e:
        print(f"  ✗ Could not remove .deduped folder - {e}")
    
    # Calculate totals
    restored_size = sum(f["size_bytes"] for f in restored_files)
    cleanup_complete = len(failed_files) == 0 and len(restored_files) == len(all_files)
    
    # Generate report
    report = {
        "action": "restore_all_files",
        "timestamp": datetime.now().isoformat() + "Z",
        "base_directory": base_dir,
        "summary": {
            "sets_processed": len(sets),
            "files_restored": len(restored_files),
            "files_failed": len(failed_files),
            "space_restored_bytes": restored_size,
            "space_restored_human": format_bytes(restored_size),
            "cleanup_complete": cleanup_complete
        },
        "restored_files": restored_files,
        "failed_files": failed_files
    }
    
    report_path = os.path.join(base_dir, args.report)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files restored: {len(restored_files)}")
    print(f"Files failed: {len(failed_files)}")
    print(f"Space restored: {format_bytes(restored_size)}")
    print(f"Cleanup complete: {cleanup_complete}")
    print(f"\nReport saved to: {report_path}")
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files could not be restored. Check report for details.")
        sys.exit(1)
    else:
        print(f"\n✓ All files have been restored to their original locations.")
        if cleanup_complete:
            print(f"✓ .deduped folder has been cleaned up.")


if __name__ == '__main__':
    main()
