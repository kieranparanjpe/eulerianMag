import math

import numpy as np
import pyaudio
import time
import wave
import csv

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
INPUT_DEVICE = 5


class Microphone:

    def __init__(self):
        self.stream = None
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.lines = []
        self.start_time = -1

        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(0, num_devices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    def start(self):
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      stream_callback=self.mic_callback,
                                      frames_per_buffer=CHUNK,
                                      input_device_index=INPUT_DEVICE)
        self.start_time = time.time()

    def mic_callback(self, input_data, frame_count, time_info, flags):
        audio_data = np.fromstring(input_data, dtype=np.int16)
        self.lines.append([time.time()-self.start_time, math.sqrt(abs(np.mean(audio_data**2))), np.mean(audio_data)])
        self.frames.append(input_data)
        return None, pyaudio.paContinue

    def write(self, filename):
        if self.start_time < 0:
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()

        with open(filename.replace(".wav", "_audio.csv"), 'w', newline='') as file:
            writer = csv.writer(file)

            for line in self.lines:
                writer.writerow(line)

    def close(self):
        if self.start_time < 0:
            return

        if self.stream is not None:
            self.stream.close()
        self.audio.terminate()
