# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Identifies and manages duplicate video files based on **content** (not filename or file hash). Detection uses audio fingerprinting as the primary method and perceptual hash (visual) as a fallback for silent videos.

## Running the tool

Dependencies are Docker-managed — run everything through Docker Compose (`compose.yml`):

```bash
# Build
docker compose build

# Scan for duplicates
docker compose exec video-dedup python src/dedup.py /videos --scan
docker compose exec video-dedup python src/dedup.py /videos --scan --dry-run
docker compose exec video-dedup python src/dedup.py /videos --scan --verbose
docker compose exec video-dedup python src/dedup.py /videos --scan --detect-upscaling
docker compose exec video-dedup python src/dedup.py /videos --scan --no-subfolders
docker compose exec video-dedup python src/dedup.py /videos --scan --include-folders path/a path/b

# Review & act on results
docker compose exec video-dedup python src/dedup.py /videos --delete           # dry run
docker compose exec video-dedup python src/dedup.py /videos --delete --confirm  # live delete
docker compose exec video-dedup python src/dedup.py /videos --restore           # full rollback
```

Syntax check (no test suite exists):
```bash
python3 -m py_compile src/dedup.py src/lib/*.py
```

## Architecture

### Multi-stage detection pipeline (`src/dedup.py`)

Pairs are filtered out early if they fail a cheaper check before reaching the expensive fingerprinting stages:

1. **File hash** — exact-match fast path
2. **Duration** — 25% tolerance (`LENGTH_TOLERANCE`); short videos (< 10 min) skip this and are compared against everything
3. **Aspect ratio** — 5% tolerance
4. **File size** — 90% tolerance (skipped for short videos)
5. **Audio fingerprint** — STFT spectral peak matching with cross-correlation; sufficient on its own
6. **Visual fingerprint** — pHash on frame clusters at 5 anchor points; fallback for silent videos

Feature extraction runs in parallel via `ProcessPoolExecutor`. Results are cached in `.dedup_cache/` (version-tracked, `CACHE_VERSION = 2` in `src/lib/config.py`).

### Library modules (`src/lib/`)

| File | Responsibility |
|------|---------------|
| `config.py` | All thresholds and constants — change behaviour here, not inline |
| `features.py` | Feature extraction, cache read/write, file discovery |
| `audio.py` | Custom STFT-based spectral fingerprinting + cross-correlation |
| `visual.py` | pHash generation via `imagehash` + Pillow |
| `quality.py` | `ffprobe` metadata, quality scoring, HDR detection, upscaling analysis |
| `utils.py` | File move/restore/delete, JSON metadata I/O |

### The `__deduped/` folder — shared data contract

All three operations (scan, delete, restore) communicate through this on-disk structure:

```
<directory>/
└── __deduped/
    └── duplicate_set_001/
        ├── video_a.mp4            # moved from original location
        ├── video_a.mp4.json       # metadata + decision
        ├── video_a.mp4.keep       # OR .delete (zero-byte marker, human-readable only)
        ├── video_b.mp4
        ├── video_b.mp4.json
        └── video_b.mp4.delete
```

Key JSON fields: `recommendation` (KEEP / DELETE_CANDIDATE), `original_full_path` (used by restore), `quality_score`, `reason`, `better_alternative`.

## Hard constraints — do not change

- `TEMP_DIR` is a module-level global set in `main()` and used by multiple functions — do not refactor into a parameter or class attribute
- `__deduped/` folder name and `duplicate_set_` prefix are hardcoded across all operations — do not rename
- Scripts are intentionally standalone — do not add `__init__.py` imports between them or turn them into a package
- Do not add `requirements.txt` or a virtualenv — Docker manages all dependencies
- `--restore` is a full rollback tool that moves back ALL files, not just KEEP — do not make it selective
