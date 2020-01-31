#!/usr/bin/env python3

import numpy as np
import pyaudio
import wave
import concurrent.futures
import fastanalyze
import time
from datetime import datetime
from dead_reckoning_script import basic_movement

def record(record_seconds, filename):
    # Setup channel info
    FORMAT = pyaudio.paInt16 # data type format
    CHANNELS = 1 # Adjust to your number of channels
    RATE = 48000 # Sample Rate
    CHUNK = 2400 # Block Size
    RECORD_SECONDS = record_seconds # Record time
    WAVE_OUTPUT_FILENAME = filename

    # Startup pyaudio instance
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(
        format=FORMAT, 
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    # Record for RECORD_SECONDS
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Write your new .wav file with built in Python 3 Wave module
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'w')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
   waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def analyze_recording(directory):
    return fastanalyze.analyze(directory)

if __name__ == "__main__":

    test_file = open("/home/pi/Desktop/startup_script/Signals/signal_log.txt",$
    ts = datetime.now()
    test_file.write("\nAudio analysis began at: " + str(ts))
    test_file.close()

    no_signal = True
    wavfiles = ["/home/pi/Desktop/startup_script/Signals/Sample_Signals/start_$

    # time.sleep(15)

sample_count = 2
while(no_signal):
    sample_count += 1
    while(no_signal):
        record(2, wavfiles[0] + str(sample_count) + ".wav")
        no_signal = analyze_recording(wavfiles[0] + str(sample_count) + ".wav")

    # no_signal = True
    # record(4, "Signals/command_signal.wav")

#    command = "move"
#    distance = -1
#    basic_movement(command, distance)
