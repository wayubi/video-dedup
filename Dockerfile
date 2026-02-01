FROM python:3.11-slim

# Install ffmpeg and dependencies
RUN apt-get update && apt-get install -y \
    tmux \
    ffmpeg \
    libsndfile1 \
    libchromaprint-dev \
    libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    ffmpeg-python \
    imagehash \
    pillow \
    numpy \
    scipy \
    tqdm \
    pyacoustid \
    audioread \
    scikit-image

# Create scripts directory and copy all scripts
RUN mkdir -p /app/scripts
COPY scripts/find_video_duplicates.py /app/scripts/
COPY scripts/dedup_delete.py /app/scripts/
COPY scripts/dedup_restore.py /app/scripts/

WORKDIR /app

# Keep container running
CMD ["tail", "-f", "/dev/null"]
