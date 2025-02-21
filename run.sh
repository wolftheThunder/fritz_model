#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Convert models if needed
python convert_models.py

# Run the server
python app.py --use_tensorrt --use_onnx \
    --model ultralight \
    --transport mediasoup \
    --mediasoup_url ws://localhost:3000 \
    --listenport 8010 