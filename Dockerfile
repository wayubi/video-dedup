FROM python:3.11-slim

# Install ffmpeg and dependencies
RUN apt-get update && apt-get install -y \
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

# Copy scripts
COPY find_video_duplicates.py /app/
COPY dedup_delete.py /app/
COPY dedup_restore.py /app/

WORKDIR /app

# Keep container running
CMD ["tail", "-f", "/dev/null"]
