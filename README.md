# Video Duplicate Detector

A Docker-based tool that identifies duplicate videos by content similarity rather than file hash. Perfect for finding the same video content encoded at different resolutions, bitrates, or with slight edits (intros, ads, etc.).

## How It Works

The tool uses a multi-stage comparison pipeline:

1. **Stage 0: File Hash** - Instant exact match for identical files
2. **Stage 1: Duration** - Fast pre-filter using duration similarity
3. **Stage 2: Aspect Ratio** - Quick visual compatibility check
4. **Stage 3: File Size** - Basic size comparison
5. **Stage 4: Audio Fingerprinting (Primary)** - Spectral fingerprint comparison
6. **Stage 5: Visual Fingerprinting (Fallback)** - Perceptual hash comparison

**Smart Comparison Strategy:**
- Short/medium videos (< 10 min) are compared against ALL videos (ensures compilations match full videos)
- Long videos (>= 10 min) use duration-based bucketing for efficiency
- Duration and file size checks are skipped when comparing two short videos
- Audio OR visual match is sufficient (either one passing = duplicate detected)

## Features

- ✅ Detects duplicates even with different encodings/resolutions
- ✅ **Cluster-based sampling** - Captures frames/samples ±10s around anchor points to handle intros
- ✅ **Smart duration handling** - Short videos compared against all; long videos use bucketing
- ✅ **Automatic fallback** - Audio primary, visual fallback for silent videos
- ✅ **Early exit** - Audio match is sufficient (visual not checked if audio passes)
- ✅ **Upscaling detection** - identifies fake 1080p/4K videos (720p upscaled)
- ✅ **Flexible folder scanning** - scan root only, all subfolders, or specific subfolders
- ✅ **Smart organization** - moves duplicates to hidden `__deduped/` folder
- ✅ **Quality scoring** - automatically ranks duplicates by quality (resolution, bitrate, codec, HDR)
- ✅ **Detailed metadata** - generates JSON metadata files with technical specs for each video
- ✅ **Intelligent recommendations** - marks best quality as "KEEP", others as "DELETE"
- ✅ **Cleanup scripts** - separate tools to safely delete candidates and restore keep files
- ✅ Generates JSON report with detailed results
- ✅ Supports MP4, MKV, AVI, MOV formats
- ✅ Progress tracking with detailed console output
- ✅ **Feature caching** - speeds up repeated scans
- ✅ **Verbose mode** - detailed comparison logs for debugging
- ✅ Dockerized for easy deployment

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository-url>
cd video-duplicate-detector
```

### 2. Edit docker-compose.yml

Update the volume mount to point to your video folder:

```yaml
volumes:
  - /path/to/your/videos:/videos  # Change left side to your video folder
```

### 3. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the container (keeps running)
docker-compose up -d

# Run the duplicate detection
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos

# Optional: dry run to preview without moving files
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --dry-run

# Optional: verbose mode for detailed comparison logs
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --verbose

# Stop the container when done
docker-compose down
```

## Command Line Options

```bash
python /app/scripts/find_video_duplicates.py /videos [OPTIONS]

Options:
  -h, --help                       Show this help message
  --dry-run                         Show duplicates without moving files
  --report FILENAME                Save JSON report (default: duplicates_report.json)
  --detect-upscaling               Analyze videos for upscaling (720p encoded as 1080p/4K)
  --include-subfolders [PATH...]   Include subfolders in scan. Without paths: all subfolders. With paths: only specified folders.
  --exclude-root                   Exclude root directory (use with --include-subfolders)
  -v, --verbose                    Show verbose output for comparison stages (debugging)
  --prune-cache                   Remove stale entries from feature cache before running
  --wipe-cache                    Delete entire feature cache before running
```

## Output Structure

After running, your directory will look like:

```
videos/
├── __deduped/                   # Hidden folder containing all duplicates
│   ├── duplicate_set_001/      # All copies of the same video
│   │   ├── video_1080p.mp4
│   │   └── video_720p_with_intro.mkv
│   └── duplicate_set_002/
│       ├── original.avi
│       └── reencoded_480p.mp4
├── unique_video.mp4            # Non-duplicates stay in place
└── duplicates_report.json      # Detailed analysis report
```

## Report Format

The JSON report contains:

```json
{
  "directory": "/videos",
  "total_videos": 150,
  "duplicate_sets": 12,
  "sets": [
    {
      "set_id": 1,
      "videos": [
        {
          "path": "/videos/movie_1080p.mp4",
          "filename": "movie_1080p.mp4",
          "duration": 7200.5
        },
        {
          "path": "/videos/movie_720p.mkv",
          "filename": "movie_720p.mkv",
          "duration": 7200.2
        }
      ]
    }
  ]
}
```

## Folder Scanning Options

The tool provides flexible folder scanning to suit different organizational needs.

### Default Behavior (Root Only)

By default, only the root directory is scanned:

```bash
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos
```

This scans only `/videos/` and ignores any subfolders.

### Scan All Subfolders

To scan the root directory plus all subfolders recursively:

```bash
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders
```

### Scan Specific Subfolders

To scan only specific subfolders (root is not included unless specified):

```bash
# Scan only the 'movies' and 'tv_shows' subfolders
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders movies tv_shows

# Scan root plus specific subfolders
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders . movies tv_shows
```

### Exclude Root Directory

To scan only subfolders and exclude the root directory:

```bash
# Scan all subfolders, excluding root
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders --exclude-root

# Scan specific subfolders only, excluding root
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders movies tv_shows --exclude-root
```

### Scan External Folders (Absolute Paths)

You can also scan folders outside the base directory by using absolute paths. This is useful when your videos are scattered across different locations.

```bash
# Scan base directory plus external folders
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /tmp/master-vids/ --include-subfolders /new/vids-a/ /new/vids-c/

# Scan only external folders (exclude base directory)
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /tmp/master-vids/ --include-subfolders /new/vids-a/ /new/vids-c/ --exclude-root

# Mix relative and absolute paths
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders ./local-subfolder /external/videos/
```

**How it works:**
- **Absolute paths** (starting with `/`): Used exactly as specified
- **Relative paths** (starting with `./`, `../`, or plain names): Resolved relative to the base directory
- All scanned videos participate in duplicate detection together
- All duplicates are organized into the base directory's `__deduped/` folder

### Examples

| Command | Scanned Locations |
|---------|-------------------|
| `python find_video_duplicates.py /videos` | Root only |
| `python find_video_duplicates.py /videos --include-subfolders` | Root + all subfolders |
| `python find_video_duplicates.py /videos --include-subfolders movies` | `movies/` subfolder only |
| `python find_video_duplicates.py /videos --include-subfolders . movies` | Root + `movies/` subfolder |
| `python find_video_duplicates.py /videos --include-subfolders --exclude-root` | All subfolders (no root) |
| `python find_video_duplicates.py /base --include-subfolders /external/vids/` | Root + external folder |
| `python find_video_duplicates.py /base --include-subfolders /ext/a/ /ext/b/ --exclude-root` | External folders only |

## How Duplicate Detection Works

### Comparison Pipeline

Videos go through a series of comparison stages. Each stage either passes or fails the comparison:

| Stage | Name | Description | Threshold |
|-------|------|-------------|-----------|
| 0 | File Hash | Exact file match | Exact match |
| 1 | Duration | Length similarity | 25% difference allowed |
| 2 | Aspect Ratio | Resolution ratio | 5% difference allowed |
| 3 | File Size | Size similarity | 90% difference allowed |
| 4 | Audio | Spectral fingerprint comparison | Per-sample threshold |
| 5 | Visual | Perceptual hash comparison | Per-frame threshold |

A video is considered a duplicate if **either** audio or visual passes (or both).

### Duration and Size Checks

- For bucket comparisons (long videos >= 10 min): Normal 25% duration and 90% size tolerances apply
- For short video comparisons (< 10 min): Duration and size checks are skipped

This allows compilations (5-10 min) and short clips (< 2 min) to match against full-length videos.

### Audio Fingerprinting

1. Extracts 5 anchor points from each video (5% to 95% of available duration)
2. For each anchor, captures 3 samples: anchor, anchor-10s, anchor+10s (15 total per video)
3. Generates spectral fingerprints using FFT analysis
4. Compares using cross-correlation with ±30s offset search
5. Match requires audio samples to pass similarity threshold

### Visual Fingerprinting (Fallback)

1. Extracts 5 anchor frames from each video (5% to 95% of available duration)
2. For each anchor, captures 3 frames: anchor, anchor-10s, anchor+10s (15 total per video)
3. Generates 3x3 region-based perceptual hashes (pHash) for each frame
4. Compares using Hamming distance
5. Match requires frames to have 3+ of 9 regions matching

### Short Video Handling

Videos under 10 minutes are compared against ALL videos (no bucketing). This ensures:
- Short clips (< 2 min) can match against long videos
- Compilations (5-10 min) can match against long videos
- Compilations can match against each other

## Verbose Mode

Use `--verbose` to see detailed comparison logs:

```bash
python find_video_duplicates.py /videos --verbose
```

Example output:
```
Comparing: video1.mp4 vs video2.mp4
  Stage 1 (Duration): diff=0.0427, threshold=0.25, result=PASS
  Stage 2 (Aspect Ratio): ratio1=1.778, ratio2=1.778, diff=0.0000, threshold=0.05, result=PASS
  Stage 3 (File Size): size1=240967540, size2=199977928, diff=0.1701, threshold=0.90, result=PASS
  Stage 4 (Audio): 5 anchor clusters
    Sample 0: anchor=74.7s
      Sample 0a: ts=74.7s vs Sample 0a: ts=32.7s: similarity=0.7142
      Sample 0a: ts=74.7s vs Sample 1a: ts=22.7s: similarity=0.6523
      ...
      Best for sample 0: sample 0a ts=32.7s similarity=0.7142 (threshold=0.8, result=FAIL)
    ...
    Audio: 3/5 anchors matched, required=2, threshold=0.8 (result=PASS)
```

## Feature Caching

The tool caches extracted features to speed up repeated scans:

```bash
# Remove stale cache entries (files that changed)
python find_video_duplicates.py /videos --prune-cache

# Delete entire cache (force re-extraction)
python find_video_duplicates.py /videos --wipe-cache
```

## Requirements

- Docker and Docker Compose
- Video formats: MP4, MKV, AVI, MOV
- Videos should be at least a few seconds long (no minimum enforced)

## Technical Details

**Built with:**
- Python 3.11
- ffmpeg (audio/video processing)
- numpy & scipy (signal processing)
- imagehash & Pillow (perceptual hashing)
- audioread (audio extraction)

**Performance:**
- Duration-based bucketing for long videos (efficiency)
- All-pairs comparison for short videos (completeness)
- Parallel feature extraction
- Feature caching

## Upscaling Detection

Videos are sometimes upscaled from lower resolutions (e.g., 720p content encoded as 4K) without actual quality improvement. This wastes storage space and provides no visual benefit.

### How to Detect Upscaled Videos

```bash
# Run with upscaling analysis
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --detect-upscaling
```

### Detection Methods

The tool uses three complementary approaches with conservative thresholds:

1. **Frequency Domain Analysis** (35% weight)
   - Analyzes 2D FFT of video frames
   - Genuine 4K/1080p has natural high-frequency detail
   - Upscaled content shows sharp frequency drop-off at original resolution
   - Most reliable indicator of upscaling

2. **Edge Sharpness Analysis** (25% weight)
   - Uses Laplacian variance to measure local sharpness
   - Upscaled edges appear softer with less detail
   - Conservative thresholds avoid flagging deliberately soft content (film grain, artistic blur)

3. **Multi-Scale Comparison** (40% weight)
   - Downsamples suspected 4K/1080p to 720p
   - Compares quality similarity
   - If downsampled version looks nearly identical, likely upscaled
   - Most accurate but computationally intensive

### Interpreting Results

**Console Output:**
```
Upscaling: 3 of 45 flagged
  - movie_4k.mp4 (3840x2160) [conf: 0.82]
  - show_upscaled.mkv (1920x1080) [conf: 0.71]
```

**JSON Report Fields:**
```json
{
  "upscaling_analysis": {
    "enabled": true,
    "threshold": 0.65,
    "summary": {
      "total_analyzed": 45,
      "upscaled_detected": 3
    },
    "videos": [
      {
        "filename": "movie_4k.mp4",
        "is_upscaled": true,
        "confidence": 0.82,
        "resolution": [3840, 2160],
        "scores": {
          "frequency": 0.75,
          "edge_sharpness": 0.60,
          "multiscale": 0.95,
          "combined": 0.82
        }
      }
    ]
  }
}
```

### Important Notes

- **Conservative by design**: Uses 0.65 confidence threshold to avoid false positives
- **Soft content safe**: Deliberately soft videos (film grain, artistic blur) won't be flagged
- **720p+ only**: Only analyzes videos ≥720p resolution
- **Confidence scores**: Individual method scores show which detection triggered
- **Not 100% accurate**: Some sophisticated upscaling may evade detection; flagged videos should be manually reviewed

## Metadata and Cleanup Scripts

After detecting duplicates, each video file gets a companion `.json` metadata file with detailed information and cleanup recommendations.

### Metadata JSON Structure

Each video in `__deduped/` gets a `{filename}.json` file:

```json
{
  "original_full_path": "/path/to/original/movie.mp4",
  "filename": "movie.mp4",
  "file_size": {"bytes": 8589934592, "human": "8.00 GB"},
  "modification_time": "2025-01-15T10:30:00Z",
  "video": {
    "width": 3840,
    "height": 2160,
    "duration_seconds": 7200,
    "duration_formatted": "02:00:00",
    "codec": "HEVC",
    "bitrate_kbps": 8000,
    "frame_rate": 24.0,
    "hdr_format": "HDR10"
  },
  "audio": [{"codec": "AAC", "bitrate_kbps": 256, "channels": 5.1, "language": "eng"}],
  "quality_score": 95,
  "recommendation": "KEEP",
  "reason": "Highest quality in set: 4K HDR"
}
```

### Delete Candidates (`dedup_delete.py`)

Safely delete all videos marked as `DELETE` from the `__deduped/` folder.

```bash
# Show what would be deleted (dry run)
docker-compose exec video-dedup python /app/scripts/dedup_delete.py /videos

# Actually delete the files
docker-compose exec video-dedup python /app/scripts/dedup_delete.py /videos --confirm
```

**Features:**
- Only deletes files with `recommendation: "DELETE"`
- Deletes both video file and its `.json` metadata
- Removes empty duplicate set folders
- Generates deletion report

**Safety:**
- Requires `--confirm` flag to actually delete
- Validates JSON and video file exist before deleting
- Reports errors without stopping

### Restore Files (`dedup_restore.py`)

Restore all videos from `__deduped/` back to their original locations.

```bash
# Restore all files to original locations
docker-compose exec video-dedup python /app/scripts/dedup_restore.py /videos

# Restore and clean up entire __deduped folder
docker-compose exec video-dedup python /app/scripts/dedup_restore.py /videos --cleanup
```

**Features:**
- Restores to `original_full_path` from metadata
- Creates parent directories if missing
- Handles filename collisions (adds `_restored_001` suffix)
- Deletes from `__deduped/` after successful restore
- Cleans up empty folders
- `--cleanup` flag removes entire `__deduped/` folder including non-dedup files

### Workflow Example

Complete workflow from detection to cleanup:

```bash
# 1. Detect duplicates
docker-compose exec video-dedup python /app/scripts/find_video_duplicates.py /videos --include-subfolders

# 2. Review __deduped/ folder and metadata JSON files
ls -la /videos/__deduped/

# 3. Preview what would be deleted (optional)
docker-compose exec video-dedup python /app/scripts/dedup_delete.py /videos

# 4. Delete candidates to free space
docker-compose exec video-dedup python /app/scripts/dedup_delete.py /videos --confirm

# 5. Restore keep files back to original locations
docker-compose exec video-dedup python /app/scripts/dedup_restore.py /videos
```

## Troubleshooting

### Container won't start
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Permission errors
Make sure your video folder is readable by the container user:
```bash
chmod -R 755 /path/to/videos
```

### Out of memory
For very large video collections, process in batches or increase Docker memory limits.

### False positives/negatives
Use `--verbose` to see detailed comparison logs. Adjust thresholds in the script if needed:
- `AUDIO_THRESHOLD`: Similarity threshold for audio samples (default: 0.8)
- `AUDIO_REQUIRED_MATCHES`: Number of anchor clusters that must match (default: 2)
- `VISUAL_FRAME_THRESHOLD`: Regions threshold for visual frames (default: 3/9)

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the need to clean up video libraries with multiple encodings
- Uses perceptual hashing techniques from image research
- Audio fingerprinting based on spectral analysis methods
