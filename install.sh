#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libavdevice-dev \
    nvidia-cuda-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p models/.onnx models/.trt web
mkdir -p data/avatars/avator_1/{full_imgs,face_imgs}

# Run setup test
python test_setup.py

echo "Installation complete. Please check test results above." 