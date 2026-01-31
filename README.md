# Video Duplicate Detector

A Docker-based tool that identifies duplicate videos by content similarity rather than file hash. Perfect for finding the same video content encoded at different resolutions, bitrates, or with slight edits (intros, ads, etc.).

## How It Works

The tool uses a two-tier approach:

1. **Audio Fingerprinting (Primary)** - Extracts audio samples from multiple points in each video and compares spectral fingerprints
2. **Visual Fingerprinting (Fallback)** - If videos lack audio, compares perceptual hashes of frames from different timestamps

**Smart Sampling Strategy:**
- Skips the first 10 seconds to avoid intros/ads
- Samples from 20%, 40%, 60%, 80% of video duration
- Groups videos by approximate duration for efficiency

## Features

- ✅ Detects duplicates even with different encodings/resolutions
- ✅ Handles 15% length differences (accounts for cuts/additions)
- ✅ Automatic fallback from audio to visual analysis
- ✅ **Upscaling detection** - identifies fake 1080p/4K videos (720p upscaled)
- ✅ Organizes duplicates into numbered folders
- ✅ Generates JSON report with detailed results
- ✅ Supports MP4, MKV, AVI, MOV formats
- ✅ Progress tracking with detailed console output
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
docker-compose exec video-dedup python /app/find_video_duplicates.py /videos

# Optional: dry run to preview without moving files
docker-compose exec video-dedup python /app/find_video_duplicates.py /videos --dry-run

# Stop the container when done
docker-compose down
```

## Command Line Options

```bash
python /app/find_video_duplicates.py /videos [OPTIONS]

Options:
  --dry-run              Show duplicates without moving files
  --report FILENAME      Save JSON report (default: duplicates_report.json)
  --detect-upscaling     Analyze videos for upscaling (720p encoded as 1080p/4K)
```

## Output Structure

After running, your directory will look like:

```
videos/
├── duplicate_set_001/          # All copies of the same video
│   ├── video_1080p.mp4
│   └── video_720p_with_intro.mkv
├── duplicate_set_002/
│   ├── original.avi
│   └── reencoded_480p.mp4
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

## How Duplicate Detection Works

### Audio Fingerprinting
1. Extracts 5-second audio samples from 4 different timestamps
2. Generates spectral fingerprints using FFT analysis
3. Compares fingerprints using correlation
4. Match requires ≥50% sample similarity

### Visual Fingerprinting (Fallback)
1. Extracts frames at 20%, 40%, 60%, 80% of duration
2. Generates perceptual hashes (pHash) for each frame
3. Compares using Hamming distance
4. Match requires ≥50% frame similarity

### Duration Matching
Videos are compared only if their durations are within 15% of each other.

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
- tqdm (progress bars)

**Performance:**
- Duration-based bucketing for efficiency
- Parallel-ready structure (samples processed sequentially per video)
- Memory-efficient temp file management

## Upscaling Detection

Videos are sometimes upscaled from lower resolutions (e.g., 720p content encoded as 4K) without actual quality improvement. This wastes storage space and provides no visual benefit.

### How to Detect Upscaled Videos

```bash
# Run with upscaling analysis
docker-compose exec video-dedup python /app/find_video_duplicates.py /videos --detect-upscaling
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
Upscaling Analysis Summary:
  Total videos analyzed: 45
  Potentially upscaled: 3

  Flagged videos:
    - movie_4k.mp4 (3840x2160) [confidence: 0.82]
    - show_upscaled.mkv (1920x1080) [confidence: 0.71]
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

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the need to clean up video libraries with multiple encodings
- Uses perceptual hashing techniques from image research
- Audio fingerprinting based on spectral analysis methods
