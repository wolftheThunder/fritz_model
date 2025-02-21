import os
import torch
import tensorrt as trt
from ultralight.unet import Model  # Your model definition

def convert_to_onnx(model_path, onnx_path):
    """Convert PyTorch model to ONNX"""
    model = Model(6, 'hubert')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dummy_input = torch.randn(1, 6, 160, 160)
    
    torch.onnx.export(model, 
                     dummy_input,
                     onnx_path,
                     opset_version=13,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    print(f"ONNX model saved to {onnx_path}")

def convert_to_tensorrt(onnx_path, trt_path):
    """Convert ONNX model to TensorRT engine"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
        
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    
    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {trt_path}")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models/.onnx", exist_ok=True)
    os.makedirs("models/.trt", exist_ok=True)
    
    # Convert models
    model_path = "models/ultralight.pth"
    onnx_path = "models/.onnx/ultralight.onnx"
    trt_path = "models/.trt/ultralight.trt"
    
    if not os.path.exists(onnx_path):
        convert_to_onnx(model_path, onnx_path)
    if not os.path.exists(trt_path):
        convert_to_tensorrt(onnx_path, trt_path) 