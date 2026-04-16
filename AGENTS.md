# video-dedup — Project Context

## What this project does
Identifies and manages duplicate video files based on **content** (not file hash).
Detection uses audio fingerprinting as the primary method and visual (perceptual hash)
fingerprinting as a fallback for videos without audio.

---

## Scripts

### `src/dedup.py` — main entrypoint (~400 lines)
Unified CLI for scanning, deleting candidates, and restoring files.

**How to run:**
```bash
python3 src/dedup.py <directory>
python3 src/dedup.py <directory> --scan
python3 src/dedup.py <directory> --scan --dry-run
python3 src/dedup.py <directory> --scan --detect-upscaling
python3 src/dedup.py <directory> --scan --no-subfolders
python3 src/dedup.py <directory> --scan --include-folders path/a path/b
python3 src/dedup.py <directory> --scan --report my_report.json
python3 src/dedup.py <directory> --delete
python3 src/dedup.py <directory> --delete --confirm
python3 src/dedup.py <directory> --restore
```

**Key internals:**
- Supported formats: `.mp4`, `.mkv`, `.avi`, `.mov`
- Uses `ffprobe` (duration) and `ffmpeg` (frame + audio extraction) via `subprocess`
- Audio fingerprint: custom STFT-based spectral peak fingerprint (no librosa dependency)
- Visual fingerprint: perceptual hash (`imagehash.phash`) on 4 frames at 20/40/60/80% of duration
- Length tolerance: 15% (`LENGTH_TOLERANCE = 0.15`) before files are considered too different to compare
- Skips first 10 seconds (`SKIP_FIRST_SECONDS`) to avoid intros/ads
- `TEMP_DIR` is a **module-level global** set in `main()` — do not refactor into a class or function parameter without updating all callers
- Upscaling detection uses three methods: `frequency_analysis`, `edge_sharpness`, `multiscale_comparison`; confidence threshold is 0.65 (conservative)

### `scripts/dedup_delete.py` — deletes candidates
Reads the `.deduped/` folder and permanently deletes files marked `DELETE_CANDIDATE`.

**Dry-run by default — requires `--confirm` to actually delete.**

```bash
python3 scripts/dedup_delete.py <directory>           # dry run
python3 scripts/dedup_delete.py <directory> --confirm  # live delete
python3 scripts/dedup_delete.py <directory> --confirm --report my_report.json
```

### `scripts/dedup_restore.py` — restores all files
Moves **all** files in `.deduped/` (not just KEEP — all of them) back to their original paths.
Handles filename collisions by appending `_restored_NNN`. Cleans up empty `duplicate_set_*`
folders and the `.deduped/` folder itself when done.

```bash
python3 scripts/dedup_restore.py <directory>
python3 scripts/dedup_restore.py <directory> --report my_report.json
```

---

## The `.deduped/` folder — core data contract

All three scripts communicate through this folder structure. Do not change it without
updating all three scripts.

```
<directory>/
└── .deduped/
    ├── duplicate_set_1/
    │   ├── video_a.mp4          # moved here from original location
    │   ├── video_a.mp4.json     # metadata + decision
    │   ├── video_a.mp4.keep     # OR .delete marker (only one exists)
    │   ├── video_b.mp4
    │   ├── video_b.mp4.json
    │   └── video_b.mp4.delete
    └── duplicate_set_2/
        └── ...
```

**JSON metadata fields** (written by `find_video_duplicates.py`, read by the other two):
- `recommendation`: `"KEEP"` or `"DELETE_CANDIDATE"` — the only field `dedup_delete.py` acts on
- `original_full_path`: absolute path before the file was moved — used by `dedup_restore.py`
- `quality_score`: numeric score used to pick the best copy
- `reason`: human-readable explanation of the recommendation
- `better_alternative`: path to the KEEP file (only on DELETE_CANDIDATE entries)
- `file_info`: dict with `modification_time` and other file metadata

**Marker files** (`.keep` / `.delete`): zero-byte files, purely for human visual inspection
of the `.deduped/` folder. The scripts read the JSON, not the markers, for decisions.

---

## External dependencies

### System (must be in Docker image — ffmpeg provides both binaries)
- `ffmpeg` — frame and audio extraction
- `ffprobe` — duration and stream probing (ships with ffmpeg)

### Python packages (must be installed via pip)
- `numpy`
- `Pillow` (PIL)
- `imagehash`
- `tqdm`
- `audioread`
- `scipy`

These are **not** part of the Python stdlib. If the container is rebuilt without them,
`find_video_duplicates.py` will fail at import time.

---

## What to avoid
- Do not refactor into a package (`__init__.py`, imports between scripts) — they are intentionally standalone
- Do not add a virtualenv or `requirements.txt` — the Docker image manages dependencies
- Do not change the `.deduped/` folder name or `duplicate_set_` prefix — hardcoded in all three scripts
- Do not make `dedup_restore.py` selective (KEEP-only) — it is intentionally a full rollback tool
- Do not rename `TEMP_DIR` or make it a function-local variable — it is checked by multiple functions
- The `__pycache__` directory is auto-generated, never edit it