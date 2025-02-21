import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorrt as trt
import onnxruntime as ort
from typing import List, Optional

class TensorRTModel:
    """TensorRT wrapper for FAN model"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Allocate device memory
        input_shape = self.engine.get_binding_shape(0)
        output_shape = self.engine.get_binding_shape(1)
        
        d_input = torch.empty(size=input_shape, dtype=torch.float32, device='cuda')
        d_output = torch.empty(size=output_shape, dtype=torch.float32, device='cuda')
        
        # Copy input to device
        d_input.copy_(x)
        
        # Execute inference
        bindings = [d_input.data_ptr(), d_output.data_ptr()]
        self.context.execute_async_v2(bindings, self.stream.cuda_stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return d_output

class ConvBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual
        return out3

# [Previous Bottleneck class implementation remains unchanged]

class HourGlass(nn.Module):
    def __init__(self, num_modules: int, depth: int, num_features: int):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self._generate_network(self.depth)

    def _generate_network(self, level: int):
        self.add_module(f'b1_{level}', ConvBlock(self.features, self.features))
        self.add_module(f'b2_{level}', ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(f'b2_plus_{level}', ConvBlock(self.features, self.features))

        self.add_module(f'b3_{level}', ConvBlock(self.features, self.features))

    def _forward(self, level: int, inp: torch.Tensor) -> torch.Tensor:
        up1 = self._modules[f'b1_{level}'](inp)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules[f'b2_{level}'](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules[f'b2_plus_{level}'](low2)

        low3 = self._modules[f'b3_{level}'](low2)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(self.depth, x)

class FAN(nn.Module):
    def __init__(self, num_modules: int = 1, use_tensorrt: bool = False, use_onnx: bool = False):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.use_tensorrt = use_tensorrt
        self.use_onnx = use_onnx

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module(f'm{hg_module}', HourGlass(1, 4, 256))
            self.add_module(f'top_m_{hg_module}', ConvBlock(256, 256))
            self.add_module(f'conv_last{hg_module}',
                          nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module(f'bn_end{hg_module}', nn.BatchNorm2d(256))
            self.add_module(f'l{hg_module}', 
                          nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    f'bl{hg_module}', 
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    f'al{hg_module}', 
                    nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0))

        # Initialize TensorRT/ONNX if needed
        if use_tensorrt:
            self.trt_model = TensorRTModel('models/fan.trt')
        elif use_onnx:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession('models/fan.onnx', providers=providers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_tensorrt:
            return [self.trt_model.forward(x)]
        elif self.use_onnx:
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: x.cpu().numpy()}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            return [torch.from_numpy(ort_outs[0]).to(x.device)]
            
        # Regular PyTorch forward pass
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        outputs = []

        for i in range(self.num_modules):
            hg = self._modules[f'm{i}'](previous)
            ll = self._modules[f'top_m_{i}'](hg)
            ll = F.relu(self._modules[f'bn_end{i}'](
                self._modules[f'conv_last{i}'](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules[f'l{i}'](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules[f'bl{i}'](ll)
                tmp_out_ = self._modules[f'al{i}'](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs

# [Previous ResNetDepth class implementation remains unchanged]