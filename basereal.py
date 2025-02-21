"""
BaseReal: Core base class for digital human streaming.
Copyright (C) 2024 LiveTalking@lipku
Licensed under the Apache License, Version 2.0.
"""

import math
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import av
from fractions import Fraction
from ttsreal import EdgeTTS, VoitsTTS, XTTS, CosyVoiceTTS, FishTTS
from tqdm import tqdm
import shutil  # <-- added to check for ffmpeg availability

def read_imgs(img_list):
    frames = []
    print("Reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class BaseReal:
    def __init__(self, opt):
        # Ensure required attributes exist on the options object.
        if not hasattr(opt, 'fps'):
            print("[WARNING] 'fps' not provided in options; using default value of 50.")
            opt.fps = 50
        if not hasattr(opt, 'customopt') or opt.customopt is None:
            opt.customopt = []
        if not hasattr(opt, 'sessionid'):
            print("[WARNING] 'sessionid' not provided in options; using default value 0.")
            opt.sessionid = 0

        self.opt = opt
        self.sessionid = opt.sessionid
        self.sample_rate = 16000
        # Compute the audio chunk size (e.g., 320 samples per 20ms at 16kHz)
        self.chunk = self.sample_rate // self.opt.fps

        # Initialize TTS based on the option (if provided).
        if hasattr(opt, 'tts'):
            if opt.tts == "edgetts":
                self.tts = EdgeTTS(opt, self)
            elif opt.tts == "gpt-sovits":
                self.tts = VoitsTTS(opt, self)
            elif opt.tts == "xtts":
                self.tts = XTTS(opt, self)
            elif opt.tts == "cosyvoice":
                self.tts = CosyVoiceTTS(opt, self)
            elif opt.tts == "fishtts":
                self.tts = FishTTS(opt, self)
            else:
                self.tts = None
        else:
            print("[WARNING] 'tts' not provided in options; TTS functionality will be disabled.")
            self.tts = None

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        # Set video dimensions from options if provided; otherwise, default to 0.
        self.width = opt.W if hasattr(opt, 'W') else 0
        self.height = opt.H if hasattr(opt, 'H') else 0
        self.curr_state = 0

        # For custom image/audio cycling
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

        print(f"BaseReal: Initialized for session {self.sessionid}, fps {opt.fps}, chunk size {self.chunk}, "
              f"video dimensions {self.width}x{self.height}.")

    def put_msg_txt(self, msg, eventpoint=None):
        """
        Called when a text message is sent (for example, an LLM reply).
        Always prints the message to the terminal.
        """
        print(f"BaseReal (session {self.sessionid}): put_msg_txt called with message: {msg}")
        if self.tts is not None:
            self.tts.put_msg_txt(msg, eventpoint)
        else:
            print("TTS not initialized; message logged.")

    def put_audio_frame(self, audio_chunk, eventpoint=None):
        """
        Forwards the audio chunk to the ASR module if available.
        """
        if hasattr(self, "asr"):
            self.asr.put_audio_frame(audio_chunk, eventpoint)
        else:
            print("ASR not available; audio frame not processed.")

    def put_audio_file(self, filebyte):
        """
        Reads an audio file from a byte stream and processes it in chunks.
        """
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk:
            self.put_audio_frame(stream[idx:idx + self.chunk])
            streamlen -= self.chunk
            idx += self.chunk

    def __create_bytes_stream(self, byte_stream):
        """
        Reads audio data from a byte stream using soundfile.
        Resamples the audio if necessary.
        """
        stream, sample_rate = sf.read(byte_stream)
        print(f"[INFO] Audio stream: sample rate {sample_rate}, shape {stream.shape}")
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            print(f"[WARN] Audio has {stream.shape[1]} channels; using the first channel.")
            stream = stream[:, 0]
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f"[WARN] Resampling from {sample_rate} to {self.sample_rate}.")
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
        return stream

    def flush_talk(self):
        """
        Clears any queued messages. This method is used to interrupt the current talk.
        """
        print(f"BaseReal (session {self.sessionid}): flush_talk called.")
        if self.tts is not None:
            self.tts.flush_talk()
        if hasattr(self, "asr"):
            self.asr.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking

    def __loadcustom(self):
        """
        Loads custom image and audio cycles from options.
        This is used for custom avatar behavior.
        """
        for item in self.opt.customopt:
            print("Loading custom option:", item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], "*.[jpJP][pnPN]*[gG]"))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def notify(self, eventpoint):
        """
        Notifies the container of an event.
        """
        print(f"BaseReal (session {self.sessionid}): notify called with eventpoint: {eventpoint}")

    def start_recording(self):
        """
        Starts recording video and audio using ffmpeg.
        Checks if ffmpeg is available; if not, prints an error and returns.
        """
        if self.recording:
            return

        # Check if ffmpeg is available in the system PATH.
        if not shutil.which("ffmpeg"):
            print("Error: ffmpeg is not installed or not found in PATH. Please install ffmpeg to enable recording.")
            return

        if self.width == 0 or self.height == 0:
            print("BaseReal: Video dimensions not set; cannot start recording.")
            return

        command = [
            "ffmpeg", "-y", "-an", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", "25", "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "h264", f"temp{self.sessionid}.mp4"
        ]
        try:
            self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
        except Exception as e:
            print(f"Error starting video recording: {e}")
            return

        acommand = [
            "ffmpeg", "-y", "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-i", "-",
            "-acodec", "aac", f"temp{self.sessionid}.aac"
        ]
        try:
            self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return

        self.recording = True
        print(f"BaseReal (session {self.sessionid}): start_recording called.")

    def record_video_data(self, image):
        """
        Writes a video frame to the ffmpeg pipe.
        Sets the video dimensions if not already set and logs the action.
        """
        if self.width == 0 or self.height == 0:
            self.height, self.width, _ = image.shape
            print(f"BaseReal (session {self.sessionid}): Video dimensions set to {self.width}x{self.height}.")
        if self.recording and self._record_video_pipe:
            try:
                self._record_video_pipe.stdin.write(image.tobytes())
                print(f"BaseReal (session {self.sessionid}): Recorded video frame.")
            except Exception as e:
                print(f"Error recording video frame: {e}")

    def record_audio_data(self, frame):
        """
        Writes an audio frame to the ffmpeg pipe and logs the action.
        """
        if self.recording and self._record_audio_pipe:
            try:
                self._record_audio_pipe.stdin.write(frame.tobytes())
                print(f"BaseReal (session {self.sessionid}): Recorded audio frame.")
            except Exception as e:
                print(f"Error recording audio frame: {e}")

    def start_custom_recording(self):
        print(f"BaseReal (session {self.sessionid}): start_custom_recording called.")

    def stop_recording(self):
        """
        Stops recording, closes the ffmpeg pipes, and combines video/audio.
        """
        if not self.recording:
            return
        self.recording = False
        if self._record_video_pipe:
            self._record_video_pipe.stdin.close()
            self._record_video_pipe.wait()
        if self._record_audio_pipe:
            self._record_audio_pipe.stdin.close()
            self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.sessionid}.aac -i temp{self.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio)
        print(f"BaseReal (session {self.sessionid}): stop_recording called. Recording saved as data/record.mp4.")

    def mirror_index(self, size, index):
        """
        Returns a mirrored index based on the cycle logic.
        """
        turn = index // size
        res = index % size
        return res if turn % 2 == 0 else size - res - 1
