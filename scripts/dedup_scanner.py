import os
from pathlib import Path
from typing import List, Optional
from dedup_config import VIDEO_EXTENSIONS


def find_videos(
    directory: str,
    include_subfolders: Optional[List[str]] = None,
    scan_mode: str = "RECURSIVE"
) -> List[str]:
    """Find all video files in directory based on scan_mode."""
    videos = []
    abs_directory = os.path.abspath(directory)
    deduped_folder = os.path.join(abs_directory, "__deduped")

    if scan_mode == "ROOT":
        for file in os.listdir(abs_directory):
            file_path = os.path.join(abs_directory, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(file_path)
    
    elif scan_mode == "TARGETED":
        if not include_subfolders:
            print("Error: Scan mode 'TARGETED' requires explicit --include-folders paths.")
            return []
        
        folders_to_scan = []
        for subfolder in include_subfolders:
            subfolder_path = (os.path.abspath(subfolder) if os.path.isabs(subfolder)
                              else os.path.abspath(os.path.join(abs_directory, subfolder)))
            
            if not os.path.exists(subfolder_path):
                print(f"Warning: Specified subfolder does not exist: {subfolder}")
                continue
            if not os.path.isdir(subfolder_path):
                print(f"Warning: Specified path is not a directory: {subfolder}")
                continue
            if subfolder_path.startswith(deduped_folder):
                print(f"Warning: Skipping __deduped folder: {subfolder_path}")
                continue
            folders_to_scan.append(subfolder_path)
        
        for folder in folders_to_scan:
            for root, _, files in os.walk(folder):
                if root.startswith(deduped_folder):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(file_path)
    
    else:  # RECURSIVE
        for root, _, files in os.walk(abs_directory):
            if root.startswith(deduped_folder):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(file_path)

    return videos