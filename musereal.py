import math
import torch
import numpy as np
import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model, load_diffusion_model, load_audio_model
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal
from tqdm import tqdm

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

@torch.no_grad()
def warm_up(batch_size, model):
    """
    Runs a warm-up pass on the MuseReal models.
    The model is fed dummy inputs to initialize weights.
    """
    print("Warming up model...")
    vae, unet, pe, timesteps, audio_processor = model
    # Create dummy whisper batch and latent batch
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)
    audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)

@torch.no_grad()
def inference(render_event, batch_size, input_latent_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue,
              vae, unet, pe, timesteps):
    """
    Inference function for MuseReal.
    While the render_event flag is set, this function repeatedly:
      - Retrieves a batch of audio features from the ASR module.
      - Processes latent inputs from a cycle of pre-computed latents.
      - Runs the diffusion model and decodes the output.
      - Places the resulting frames (along with associated audio frames) into the result queue.
    """
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print("Starting inference in MuseReal...")
    while render_event.is_set():
        try:
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence = True
        audio_frames = []
        # Collect audio frames (2 per batch item, adjust if needed)
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
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length, index + i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            counttime += (time.time() - t)
            count += batch_size
            if count >= 100:
                print(f"Actual avg infer fps: {count/counttime:.4f}")
                count = 0
                counttime = 0
            for i, res_frame in enumerate(recon):
                res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
    print("MuseReal inference processor stopped.")

#############################
# MuseReal Class
#############################

class MuseReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        """
        Initializes MuseReal with:
          - opt: options namespace (must include at least: fps, batch_size, W, H, avatar_id)
          - model: a tuple (vae, unet, pe, timesteps, audio_processor)
          - avatar: a tuple containing cycles for:
              (frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle)
        """
        # Ensure required options exist; assign defaults if missing.
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

        # Call the BaseReal constructor (which may also compute audio chunks using opt.fps)
        super().__init__(opt)
        self.sessionid = getattr(opt, "sessionid", "unknown")
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps
        self.batch_size = opt.batch_size
        self.idx = 0
        # Using a multiprocessing Queue for inter-thread communication
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        # Unpack the diffusion and audio model tuple
        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model
        (self.frame_list_cycle, self.mask_list_cycle,
         self.coord_list_cycle, self.mask_coords_list_cycle,
         self.input_latent_list_cycle) = avatar
        # Initialize ASR with the audio processor (MuseASR must be implemented accordingly)
        self.asr = MuseASR(opt, self, self.audio_processor)
        self.asr.warm_up()
        # Create an event flag for controlling inference threads
        self.render_event = mp.Event()
        self.render_event.set()  # Start with render event set to True
        self.speaking = False
        # For custom video handling (if implemented)
        self.custom_index = {}
        self.custom_img_cycle = {}

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Processes frames produced by the inference thread.
        Combines the rendered image with the original full frame (or a custom cycle) based on audio state,
        then packages video and audio frames into the appropriate tracks.
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            # If both audio frames indicate non-silence, use the full frame cycle
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
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                # Expecting bbox in the form (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
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
        print("MuseReal process_frames thread stopped.")

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Starts the inference and frame processing threads.
        Also continuously runs ASR steps until quit_event is set.
        """
        if hasattr(self, "tts"):
            self.tts.render(quit_event)
        else:
            print("TTS not available; skipping.")
        if hasattr(self, "init_customindex"):
            self.init_customindex()
        else:
            print("Custom index init not defined; skipping.")
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()
        Thread(target=inference, args=(self.render_event, self.batch_size, self.input_latent_list_cycle,
                                        self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue,
                                        self.vae, self.unet, self.pe, self.timesteps)).start()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        self.render_event.clear()
        print("MuseReal thread stopped.")

    # Container methods (replace prints with your actual business logic as needed)
    def put_msg_txt(self, msg):
        print(f"MuseReal({self.sessionid}): Received message -> {msg}")

    def notify(self, eventpoint):
        print(f"MuseReal({self.sessionid}): Received event -> {eventpoint}")

    def flush_talk(self):
        print(f"MuseReal({self.sessionid}): flush_talk executed.")

    def set_curr_state(self, audiotype, reinit):
        print(f"MuseReal({self.sessionid}): set_curr_state called with audiotype={audiotype}, reinit={reinit}")

    def start_recording(self):
        print(f"MuseReal({self.sessionid}): start_recording executed.")

    def stop_recording(self):
        print(f"MuseReal({self.sessionid}): stop_recording executed.")

    def is_speaking(self):
        return self.speaking

    def put_audio_file(self, filebytes):
        print(f"MuseReal({self.sessionid}): put_audio_file received {len(filebytes)} bytes.")
