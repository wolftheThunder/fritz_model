# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libavdevice-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/.onnx models/.trt web \
    && mkdir -p data/avatars/avator_1/{full_imgs,face_imgs}

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8010

# Run the application
CMD ["python3", "app.py", "--use_tensorrt", "--use_onnx", "--model", "ultralight", "--transport", "mediasoup"] 