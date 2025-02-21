import math
import torch
import numpy as np
import os
import time
import cv2
import glob
import pickle
import copy
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal
from tqdm import tqdm
import tensorrt as trt
import onnxruntime as ort

# Additional imports for LightReal
from ultralight.unet import Model  # Your custom model class for LightReal
from ultralight.audio2feature import Audio2Feature  # For extracting audio features
from hubertasr import HubertASR  # Your ASR module for LightReal
from transformers import Wav2Vec2Processor, HubertModel  # If used in your processing

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for inference.")

#############################
# Helper Functions
#############################

def read_imgs(img_list):
    frames = []
    print("Reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1

#############################
# Model and Avatar Loading Functions
#############################

def load_model(opt, use_tensorrt=True):
    """Load model with TensorRT or ONNX optimization"""
    model_name = "ultralight"
    model_path = f"./models/{model_name}.pth"
    onnx_path = f"./models/.onnx/{model_name}.onnx"
    trt_path = f"./models/.trt/{model_name}.trt"

    if use_tensorrt and os.path.exists(trt_path):
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    elif os.path.exists(onnx_path):
        # Load ONNX model
        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        return session
    else:
        # Load PyTorch model
        model = Model(6, 'hubert').to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        return model.eval()

def load_avatar(avatar_id):
    """
    Loads the avatar images and coordinates.
    Expects the following folder structure under ./data/avatars/{avatar_id}:
      - full_imgs/ : Full-body images
      - face_imgs/ : Cropped face images
      - coords.pkl : Pickle file with face coordinates
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    
    # Load the coordinate cycle
    with open(coords_path, "rb") as f:
        coord_list_cycle = pickle.load(f)
    # Load full-body images
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]")),
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    # Load face images
    input_face_list = sorted(glob.glob(os.path.join(face_imgs_path, "*.[jpJP][pnPN]*[gG]")),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    return frame_list_cycle, face_list_cycle, coord_list_cycle

def warm_up(batch_size, avatar, modelres):
    """Warm up the model"""
    model = avatar
    if isinstance(model, trt.ICudaEngine):
        # TensorRT warmup
        context = model.create_execution_context()
        input_shape = (batch_size, 6, modelres, modelres)
        dummy_input = np.ones(input_shape, dtype=np.float32)
        output_shape = (batch_size, 3, modelres, modelres)
        output = np.empty(output_shape, dtype=np.float32)
        context.execute_v2([dummy_input, output])
    elif isinstance(model, ort.InferenceSession):
        # ONNX Runtime warmup
        dummy_input = np.ones((batch_size, 6, modelres, modelres), dtype=np.float32)
        model.run(None, {'input': dummy_input})
    else:
        # PyTorch warmup
        dummy_input = torch.ones(batch_size, 6, modelres, modelres).to(device)
        with torch.no_grad():
            model(dummy_input)

#############################
# Inference Function
#############################

def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print("Starting inference in LightReal...")
    while not quit_event.is_set():
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size * 2):
            frame, type_val, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, type_val, eventpoint))
            if type_val == 0:
                is_all_silence = False
        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
        else:
            t = time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length, index + i)
                # For LightReal, process face images with cropping and masking.
                crop_img = face_list_cycle[idx]
                # Crop region (example values; adjust based on your use-case)
                crop_img_ex = crop_img[4:164, 4:164].copy()
                # Create a masked version by drawing a black rectangle (example logic)
                img_masked = crop_img_ex.copy()
                cv2.rectangle(img_masked, (5,5), (150,145), (0,0,0), -1)
                # Transpose channels for PyTorch (from HWC to CHW)
                img_real = crop_img_ex.transpose(2,0,1).astype(np.float32)
                img_masked = img_masked.transpose(2,0,1).astype(np.float32)
                # Concatenate masked and real images along the channel axis
                img_concat = np.concatenate([img_real, img_masked], axis=0)[None]
                img_batch.append(torch.from_numpy(img_concat / 255.0))
            img_batch = torch.stack(img_batch).to(device)
            reshaped_mel_batch = [arr.reshape(32, 32, 32) for arr in mel_batch]
            mel_batch = torch.stack([torch.from_numpy(arr) for arr in reshaped_mel_batch]).to(device)
            with torch.no_grad():
                pred = model(img_batch, mel_batch)
            pred = pred.cpu().numpy().transpose(0,2,3,1) * 255.
            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                print(f"Actual avg infer fps: {count/counttime:.4f}")
                count = 0
                counttime = 0
            for i, res_frame in enumerate(pred):
                res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
    print("LightReal inference processor stopped.")

#############################
# LightReal Class
#############################

class LightReal(BaseReal):
    def __init__(self, opt, model_avatar):
        """
        Expects model_avatar as a tuple: (model, frame_list_cycle, face_list_cycle, coord_list_cycle)
        """
        # Ensure required attributes are set on opt.
        if not hasattr(opt, 'fps'):
            print("[WARNING] 'fps' not provided in options; using default value of 50.")
            opt.fps = 50
        if not hasattr(opt, 'batch_size'):
            print("[WARNING] 'batch_size' not provided in options; using default value of 1.")
            opt.batch_size = 1
        if not hasattr(opt, 'W'):
            print("[WARNING] 'W' (width) not provided in options; using default value of 640.")
            opt.W = 640
        if not hasattr(opt, 'H'):
            print("[WARNING] 'H' (height) not provided in options; using default value of 480.")
            opt.H = 480
        if not hasattr(opt, 'avatar_id'):
            print("[WARNING] 'avatar_id' not provided in options; using default value of 'default'.")
            opt.avatar_id = "default"

        super().__init__(opt)
        self.sessionid = getattr(opt, "sessionid", "unknown")
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size * 2)
        self.model, self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = model_avatar
        # Note: load_model(opt)[1] returns the audio processor (Audio2Feature) used by ASR.
        self.asr = HubertASR(opt, self, load_model(opt)[1])
        self.asr.warm_up()
        self.render_event = mp.Event()
        self.speaking = False
        # For custom video handling; set these up if available.
        self.custom_index = {}
        self.custom_img_cycle = {}

    def inference(self, input_data):
        """Run inference using the appropriate backend"""
        if isinstance(self.model, trt.ICudaEngine):
            # TensorRT inference
            context = self.model.create_execution_context()
            output = np.empty((self.batch_size, 3, 160, 160), dtype=np.float32)
            context.execute_v2([input_data, output])
            return output
        elif isinstance(self.model, ort.InferenceSession):
            # ONNX Runtime inference
            output = self.model.run(None, {'input': input_data})[0]
            return output
        else:
            # PyTorch inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_data).to(device)
                output = self.model(input_tensor)
                return output.cpu().numpy()

    def process_frame(self, face_frame):
        """Process a single frame"""
        # Prepare input
        face_input = cv2.resize(face_frame, (160, 160))
        face_input = face_input.transpose(2, 0, 1).astype(np.float32) / 255.0
        face_input = np.expand_dims(face_input, 0)
        
        # Run inference
        output = self.inference(face_input)
        
        # Post-process output
        output = (output.transpose(0, 2, 3, 1) * 255).astype(np.uint8)[0]
        return output

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None:
                    mirindex = self.custom_index[audiotype] % len(self.custom_img_cycle[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    crop_img = self.face_list_cycle[idx]
                    crop_img[4:164, 4:164] = res_frame.astype(np.uint8)
                    crop_img = cv2.resize(crop_img, (x2 - x1, y2 - y1))
                except Exception:
                    continue
                combine_frame[y1:y2, x1:x2] = crop_img
            new_frame = VideoFrame.from_ndarray(combine_frame, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame, None)), loop)
            self.record_video_data(combine_frame)  # Actual recording logic in BaseReal or elsewhere
            for audio_frame in audio_frames:
                frame, type_val, eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame, eventpoint)), loop)
                self.record_audio_data(frame)
        print("LightReal process_frames thread stopped.")

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        if hasattr(self, "tts"):
            self.tts.render(quit_event)
        else:
            print("TTS not available; skipping.")
        if hasattr(self, "init_customindex"):
            self.init_customindex()
        else:
            print("Custom index initialization not defined; skipping.")
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()
        Thread(target=inference, args=(quit_event, self.batch_size, self.face_list_cycle,
                                        self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue,
                                        self.model)).start()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        print("LightReal thread stopped.")

    # Actual container methods with real logic hooks (replace these prints with your business logic if available)
    def put_msg_txt(self, msg):
        print(f"LightReal({self.sessionid}): Received message -> {msg}")

    def notify(self, eventpoint):
        print(f"LightReal({self.sessionid}): Received event -> {eventpoint}")

    def flush_talk(self):
        print(f"LightReal({self.sessionid}): flush_talk called.")

    def set_curr_state(self, audiotype, reinit):
        print(f"LightReal({self.sessionid}): set_curr_state called with audiotype={audiotype}, reinit={reinit}")

    def start_recording(self):
        print(f"LightReal({self.sessionid}): start_recording called.")

    def stop_recording(self):
        print(f"LightReal({self.sessionid}): stop_recording called.")

    def is_speaking(self):
        return self.speaking

    def put_audio_file(self, filebytes):
        print(f"LightReal({self.sessionid}): put_audio_file received {len(filebytes)} bytes.")
