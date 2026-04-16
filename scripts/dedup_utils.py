import os
import json
import shutil
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
    """Scan __deduped folder for all duplicate set folders."""
    deduped_path = os.path.join(base_dir, "__deduped")
    
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
        if 'recommendation' not in metadata:
            return None
        return metadata
    except Exception:
        return None


def get_associated_video_path(json_path: str) -> Optional[str]:
    """Get the path to the video file associated with a JSON file."""
    if json_path.endswith('.json'):
        video_path = json_path[:-5]
        if os.path.exists(video_path):
            return video_path
    return None


def find_delete_files(set_path: str) -> List[Tuple[str, str, str, Dict]]:
    """Find all DELETE files in a duplicate set."""
    candidates = []
    
    for file in os.listdir(set_path):
        if file.endswith('.json'):
            json_path = os.path.join(set_path, file)
            metadata = load_video_metadata(json_path)
            
            if metadata and metadata.get('recommendation') == 'DELETE':
                video_path = get_associated_video_path(json_path)
                if video_path:
                    marker_path = video_path + '.delete' if os.path.exists(video_path + '.delete') else None
                    candidates.append((video_path, json_path, marker_path, metadata))
    
    return candidates


def find_all_files(set_path: str) -> List[Tuple[str, str, Optional[str], Dict]]:
    """Find all files in a duplicate set."""
    all_files = []
    
    for file in os.listdir(set_path):
        if file.endswith('.json'):
            json_path = os.path.join(set_path, file)
            metadata = load_video_metadata(json_path)
            
            if metadata:
                video_path = get_associated_video_path(json_path)
                if video_path:
                    marker_path = None
                    if os.path.exists(video_path + '.keep'):
                        marker_path = video_path + '.keep'
                    elif os.path.exists(video_path + '.delete'):
                        marker_path = video_path + '.delete'
                    all_files.append((video_path, json_path, marker_path, metadata))
    
    return all_files


def delete_file_pair(video_path: str, json_path: str, marker_path: Optional[str] = None) -> Tuple[bool, str]:
    """Delete a video file, its associated JSON, and marker file."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        if marker_path and os.path.exists(marker_path):
            os.remove(marker_path)
        return True, ""
    except Exception as e:
        return False, str(e)


def restore_file(video_path: str, json_path: str, marker_path: Optional[str], metadata: Dict) -> Tuple[bool, str, str, bool]:
    """Restore a video file to its original location."""
    try:
        original_path = metadata.get('original_full_path')
        if not original_path:
            return False, "", "No original path in metadata", False
        
        parent_dir = os.path.dirname(original_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        destination = original_path
        collision_handled = False
        
        if os.path.exists(destination):
            base_name, ext = os.path.splitext(original_path)
            counter = 1
            while os.path.exists(destination):
                destination = f"{base_name}_restored_{counter:03d}{ext}"
                counter += 1
            collision_handled = True
        
        shutil.move(video_path, destination)
        
        if os.path.exists(json_path):
            os.remove(json_path)
        if marker_path and os.path.exists(marker_path):
            os.remove(marker_path)
        
        return True, destination, "", collision_handled
        
    except Exception as e:
        return False, "", str(e), False


def cleanup_empty_deduped_folders(base_dir: str) -> Tuple[List[str], List[str]]:
    """Clean up empty duplicate set folders. Returns (removed, skipped)."""
    removed = []
    skipped = []
    
    sets = scan_deduped_folders(base_dir)
    for set_path in sets:
        remaining = [f for f in os.listdir(set_path) if os.path.isfile(os.path.join(set_path, f))]
        if not remaining:
            try:
                os.rmdir(set_path)
                removed.append(os.path.basename(set_path))
            except Exception:
                pass
        else:
            skipped.append(os.path.basename(set_path))
    
    deduped_path = os.path.join(base_dir, "__deduped")
    if os.path.exists(deduped_path) and not os.listdir(deduped_path):
        try:
            os.rmdir(deduped_path)
        except Exception:
            pass
    
    return removed, skipped