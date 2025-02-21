from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
import tensorrt as trt
import onnxruntime as ort
from typing import Optional, List, Tuple, Union

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *

class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3

class NetworkSize(Enum):
    LARGE = 4
    
    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class TRTEngine:
    """TensorRT engine wrapper"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(self.engine.get_binding_shape(1).nbytes)
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, input_data, stream)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )
        
        # Transfer results back
        output = np.empty(self.engine.get_binding_shape(1), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        
        # Synchronize
        stream.synchronize()
        
        return output

class FaceAlignment:
    def __init__(self, 
                 landmarks_type, 
                 network_size=NetworkSize.LARGE,
                 device='cuda', 
                 flip_input=False, 
                 face_detector='sfd', 
                 verbose=False,
                 use_tensorrt=False,
                 use_onnx=False):
        
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        self.use_tensorrt = use_tensorrt
        self.use_onnx = use_onnx

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Initialize face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                        globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

        # Initialize model based on optimization choice
        if use_tensorrt:
            self._init_tensorrt()
        elif use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch(network_size)

    def _init_tensorrt(self):
        """Initialize TensorRT engine"""
        engine_path = os.path.join(ROOT, 'models', 'face_alignment.trt')
        if not os.path.exists(engine_path):
            # Convert PyTorch model to TensorRT
            self._convert_to_tensorrt(engine_path)
        self.model = TRTEngine(engine_path)

    def _init_onnx(self):
        """Initialize ONNX Runtime session"""
        onnx_path = os.path.join(ROOT, 'models', 'face_alignment.onnx')
        if not os.path.exists(onnx_path):
            # Convert PyTorch model to ONNX
            self._convert_to_onnx(onnx_path)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = ort.InferenceSession(onnx_path, providers=providers)

    def _init_pytorch(self, network_size):
        """Initialize PyTorch model"""
        network_size = int(network_size)
        self.model = FAN(network_size)
        self.model.to(self.device)
        self.model.eval()

    def _convert_to_tensorrt(self, engine_path):
        """Convert PyTorch model to TensorRT"""
        import torch2trt
        
        model = FAN(int(NetworkSize.LARGE))
        model.eval().cuda()
        
        # Create dummy input
        x = torch.randn(1, 3, 256, 256).cuda()
        
        # Convert to TensorRT
        model_trt = torch2trt.torch2trt(
            model,
            [x],
            fp16_mode=True,
            max_workspace_size=1<<25
        )
        
        torch.save(model_trt.state_dict(), engine_path)

    def _convert_to_onnx(self, onnx_path):
        """Convert PyTorch model to ONNX"""
        model = FAN(int(NetworkSize.LARGE))
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 3, 256, 256)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )

    def get_detections_for_batch(self, images: np.ndarray) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Detect faces in a batch of images
        Args:
            images: Batch of images (B, H, W, C)
        Returns:
            List of face detections (x1, y1, x2, y2) or None if no face detected
        """
        # Convert BGR to RGB
        images = images[..., ::-1]
        
        # Get face detections
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results

    def get_landmarks_from_batch(self, images: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Get facial landmarks for batch of images using optimized inference
        """
        if self.use_tensorrt:
            return self._get_landmarks_tensorrt(images)
        elif self.use_onnx:
            return self._get_landmarks_onnx(images)
        else:
            return self._get_landmarks_pytorch(images)

    def _get_landmarks_tensorrt(self, images: np.ndarray) -> List[Optional[np.ndarray]]:
        """TensorRT optimized landmark detection"""
        # Preprocess images
        images = self._preprocess_images(images)
        
        # Run inference
        landmarks = self.model.infer(images)
        
        # Post-process results
        return self._postprocess_landmarks(landmarks)

    def _get_landmarks_onnx(self, images: np.ndarray) -> List[Optional[np.ndarray]]:
        """ONNX Runtime optimized landmark detection"""
        # Preprocess images
        images = self._preprocess_images(images)
        
        # Run inference
        landmarks = self.model.run(None, {'input': images})[0]
        
        # Post-process results
        return self._postprocess_landmarks(landmarks)

    def _get_landmarks_pytorch(self, images: np.ndarray) -> List[Optional[np.ndarray]]:
        """PyTorch landmark detection"""
        with torch.no_grad():
            # Preprocess images
            images = torch.from_numpy(self._preprocess_images(images)).to(self.device)
            
            # Run inference
            landmarks = self.model(images)
            
            # Post-process results
            return self._postprocess_landmarks(landmarks.cpu().numpy())

    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images for inference"""
        # Add preprocessing steps here
        return images

    def _postprocess_landmarks(self, landmarks: np.ndarray) -> List[Optional[np.ndarray]]:
        """Post-process landmark predictions"""
        # Add postprocessing steps here
        return landmarks