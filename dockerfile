FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch 2.8.0 with CUDA 12.9 (works with RTX 5080!)
RUN pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install Ultralytics and dependencies
RUN pip3 install --break-system-packages ultralytics opencv-python-headless numpy python-dotenv

# Copy inference script
COPY inference.py .

ENV PROCESS_FUNCTION_KEY=im17DX93TP7N48ez24JXi2Zc2KO7mTdH6DI4d7KV5E6xAzFuThfqeA==

# Create directories
RUN mkdir -p /app/input /app/output /app/weights
RUN mkdir -p /app/input/Station3-1 /app/input/Station3-2 /app/input/Station3-3

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python3", "inference.py"]