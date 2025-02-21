import torch
import tensorrt as trt
import onnxruntime as ort
import os

def test_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

def test_models():
    model_paths = {
        "PyTorch": "models/ultralight.pth",
        "ONNX": "models/.onnx/ultralight.onnx",
        "TensorRT": "models/.trt/ultralight.trt"
    }
    
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            print(f"{model_type} model found at: {path}")
        else:
            print(f"{model_type} model missing: {path}")

def test_versions():
    print("\nVersion Information:")
    print(f"PyTorch: {torch.__version__}")
    print(f"TensorRT: {trt.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")

if __name__ == "__main__":
    print("Testing CUDA Setup...")
    test_cuda()
    
    print("\nChecking Model Files...")
    test_models()
    
    print("\nChecking Library Versions...")
    test_versions() 