#!/bin/bash

# Stop any running container
docker rm -f yolov8-inference 2>/dev/null

# Build image with CUDA support (no cache to ensure fresh PyTorch installation)
echo "🔨 Building Docker image with CUDA support..."
docker build --no-cache -t yolov8-inference .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build complete!"
echo ""

# Verify CUDA is available in the image
echo "🔍 Verifying CUDA support..."
docker run --rm --gpus all yolov8-inference python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo ""
echo "🚀 Starting inference container..."

# Run container with GPU support
docker run --rm -it \
  --gpus all \
  --name yolov8-inference \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /mnt/photos:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/weights:/app/weights:ro \
  yolov8-inference \
  --weights /app/weights/best.pt \
  --source /app/input \
  --conf 0.78 \
  --device auto