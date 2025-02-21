#!/bin/bash

# Convert models if needed
python3 convert_models.py

# Run tests
python3 test_setup.py

# Start the application
exec python3 app.py \
    --use_tensorrt \
    --use_onnx \
    --model ultralight \
    --transport mediasoup \
    --mediasoup_url ${MEDIASOUP_URL} \
    --listenport 8010 