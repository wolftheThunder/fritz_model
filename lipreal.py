#!/usr/bin/env python3
"""
LipReal: Container class for Wav2Lip digital human streaming.
This module loads the Wav2Lip model and avatar, warms up the model, and
handles processing of text messages (including LLM replies), video, and audio.
"""

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
from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal
from tqdm import tqdm

# Set device for inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for inference.")

#############################
# Model and Avatar Loading
#############################

def _load(checkpoint_path):
    return torch.load(checkpoint_path) if device == 'cuda' else torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

def load_model(path):
    """
    Loads the Wav2Lip model from the given checkpoint path.
    Removes any "module." prefixes from the state dict keys.
    """
    model = Wav2Lip()
    print(f"Loading checkpoint from: {path}")
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(new_s)
    return model.to(device).eval()

def load_avatar(avatar_id):
    """
    Loads avatar images and coordinates from the expected folder structure.
    Expects:
      - full_imgs/ : Full-body images
      - face_imgs/ : Cropped face images
      - coords.pkl : Pickle file with face coordinates
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]")),
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = sorted(glob.glob(os.path.join(face_imgs_path, "*.[jpJP][pnPN]*[gG]")),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    return frame_list_cycle, face_list_cycle, coord_list_cycle

def warm_up(batch_size, model, modelres):
    """
    Runs a warm-up pass on the model.
    The model is fed dummy image and mel batches.
    """
    print("Warming up model...")
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)
    print("Model warm-up complete.")

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
# Inference Function
#############################

def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    """
    Inference function to process audio and generate video frames.
    This function simulates processing by feeding image and mel batches to the model.
    """
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print("Starting inference in LipReal...")
    while not quit_event.is_set():
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence = True
        audio_frames = []
        # Collect 2 audio frames per batch (adjust if needed)
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
            t = time.time()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length, index + i)
                face = face_list_cycle[idx]
                img_batch.append(face)
            # Convert lists to numpy arrays for processing
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            # Create a masked version of the image batch (example: mask bottom half)
            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0
            # Concatenate along the channel dimension
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            # Reshape mel batch to add a channel dimension
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            # Convert to torch tensors with proper channel order: (N, C, H, W)
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            counttime += (time.time() - t)
            count += batch_size
            if count >= 100:
                print(f"Actual avg infer fps: {count/counttime:.4f}")
                count = 0
                counttime = 0
            for i, res_frame in enumerate(pred):
                res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
    print("LipReal inference processor stopped.")

#############################
# LipReal Class
#############################

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        # Ensure required options exist; set defaults if missing.
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

        # Call the BaseReal constructor.
        super().__init__(opt)
        self.sessionid = getattr(opt, "sessionid", "unknown")
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size * 2)
        self.model = model
        # Unpack avatar data.
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        # Initialize ASR module.
        self.asr = LipASR(opt, self)
        self.asr.warm_up()
        self.render_event = mp.Event()
        self.speaking = False
        # For custom video handling.
        self.custom_index = {}
        self.custom_img_cycle = {}

        print(f"LipReal({self.sessionid}): Initialized with model and avatar.")

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Processes inference results and feeds video/audio tracks.
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            # If both audio frames indicate non-silence, use the full frame cycle.
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:
                self.speaking = False
                audiotype = audio_frames[0][1]
                if hasattr(self, "custom_index") and self.custom_index.get(audiotype) is not None:
                    mirindex = self.custom_index[audiotype] % len(self.custom_img_cycle[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                # Create a copy of the current full frame.
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                # Expect bbox in the form (y1, y2, x1, x2)
                y1, y2, x1, x2 = bbox
                try:
                    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    continue
                combine_frame[y1:y2, x1:x2] = res_frame_resized
            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame, None)), loop)
            self.record_video_data(image)
            for audio_frame in audio_frames:
                frame, type_val, eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame, eventpoint)), loop)
                self.record_audio_data(frame)
        print("LipReal process_frames thread stopped.")

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Starts threads for processing frames and runs ASR continuously.
        """
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
                                        self.model,)).start()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        print("LipReal thread stopped.")

    def put_msg_txt(self, msg):
        """
        Receives a text message (e.g. an LLM reply) and prints it.
        """
        print(f"LipReal({self.sessionid}): Received message -> {msg}")

    def notify(self, eventpoint):
        print(f"LipReal({self.sessionid}): Received event -> {eventpoint}")

    def flush_talk(self):
        print(f"LipReal({self.sessionid}): flush_talk called.")

    def set_curr_state(self, audiotype, reinit):
        print(f"LipReal({self.sessionid}): set_curr_state -> audiotype={audiotype}, reinit={reinit}")

    def start_recording(self):
        print(f"LipReal({self.sessionid}): start_recording called.")

    def stop_recording(self):
        print(f"LipReal({self.sessionid}): stop_recording called.")

    def is_speaking(self):
        return self.speaking

    def put_audio_file(self, filebytes):
        print(f"LipReal({self.sessionid}): put_audio_file received {len(filebytes)} bytes.")
