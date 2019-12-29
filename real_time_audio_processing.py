import numpy as np
import pyaudio
import wave
import concurrent.futures


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
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def analyze_recording():
    return false

if __name__ == "__main__":
    no_signal = True

    while(no_signal):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(analyze_recording)
            no_signal = future.result()
        record(2, "Signals/start_signal.py")

    record(4, "Signals/command_signal.py")
    ## --------------------------------
    ## Determine the command here
    ## --------------------------------
        