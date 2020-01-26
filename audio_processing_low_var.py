from __future__ import division
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_widths
from teager_py import Teager
from math import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate
from time import time
import sys

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT
    
    Pros: Accurate, usually even more so than zero crossing counter 
    (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic interpolation 
    being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length
    
    Cons: Doesn't find the right value if harmonics are stronger than 
    fundamental, which is common.
    
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]
    
    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def freq_from_crossings(sig, fs):
    """Estimate frequency by counting zero crossings
    
    Pros: Fast, accurate (increasing with signal length).  Works well for long 
    low-noise sines, square, triangle, etc.
    
    Cons: Doesn't work if there are multiple zero crossings per cycle, 
    low-frequency baseline shift, noise, etc.
    
    """
    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    #crossings = indices
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better. Spline, cubic, whatever
    
    return fs / mean(diff(crossings))

# # Setup channel info
# FORMAT = pyaudio.paInt16 # data type format
# CHANNELS = 1 # Adjust to your number of channels
# RATE = 44100 # Sample Rate
# CHUNK = 44100 # Block Size
# RECORD_SECONDS = 5 # Record time
# WAVE_OUTPUT_FILENAME = "FM_Samples/dynamic_samples/10600_to_12600_step_up_0005sec_88200_5.wav"

# # Startup pyaudio instance
# audio = pyaudio.PyAudio()

# # start Recording
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                 rate=RATE, input=True,
#                 frames_per_buffer=CHUNK)
# print ("recording...")
# frames = []

# # Record for RECORD_SECONDS
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
# print ("finished recording")

# # Stop Recording
# stream.stop_stream()
# stream.close()
# audio.terminate()

# # Write your new .wav file with built in Python 3 Wave module
# waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# waveFile.setnchannels(CHANNELS)
# waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# waveFile.setframerate(RATE)
# waveFile.writeframes(b''.join(frames))
# waveFile.close()


# Procedure:
# 1. Filter for different amplitudes
# 2. Filter for different frequencies on each carrier
# (Carrier changes on a peak-by-peak basis)
# 3. Decode into bits


# fs, data = wavfile.read("/media/ian/Elements/Data/wakeupsignal_11000_12000_900.wav")
fs, data = wavfile.read("/media/ian/Elements/Data/piecewise_10600_to_12600_780.wav")


data = [np.int64(i) for i in data]

# Teager
# !IMPORTANT
# Consider reversing the data set then performing the teager operator again
# Compare the results of both to get middle values
teager_first_data = Teager(np.abs(data), 'horizontal', 1)
teager_first_data = [np.int64(i) for i in teager_first_data]

teager_second_data = Teager(np.abs(teager_first_data), 'horizontal', 1)
teager_second_data = [np.int64(i) for i in teager_second_data]

max_value = np.amax(teager_second_data)
# This might need to be improved but does work well currently
post_processed = [i if i > 1/100*max_value else 0 for i in teager_second_data]

# Find Peaks
# !IMPORTANT
# Use this function multiple times to get different bit ranges
# 1000 picks up i % 4 very well while 500 picks up i % 2 very well
# spacing between bits at rate of 100 bits/s is approximately 250: 
# 0 --> 1 = distance of 250
peaks, _ = find_peaks(post_processed, None, 0, 1, 1/20*max_value, [1.25, 2])
peak_dimensions = peak_widths(post_processed, peaks, rel_height=0.5)

# print(peaks[0:100])
# print(peaks[0])
# print(peaks[1])
# print(peak_dimensions[0])

chunk_list = []
post_processed_data_chunk = []
for i in range(len(peaks)):
    if len(chunk_list) == 0:
        chunk_list.append(peaks[i])
    elif peaks[i] - peaks[i - 1] < 400:
        chunk_list.append(peaks[i])
        if i == len(peaks) - 1:
            post_processed_data_chunk.append(chunk_list)
    else:
        post_processed_data_chunk.append(chunk_list)
        chunk_list = []
        chunk_list.append(peaks[i])
        if i == len(peaks) - 1:
            post_processed_data_chunk.append(chunk_list)

post_processed_data_chunk = [floor(np.mean(i)) for i in post_processed_data_chunk]

# print(len(post_processed_data_chunk))

post_processed_data = [1 for i in teager_second_data]
for i in range(len(teager_second_data)):
    if i not in post_processed_data_chunk:
        post_processed_data[i] = 0

peaks, _ = find_peaks(post_processed_data, None, 0, 1, 1)

# Find the carrier frequency
# print(data[peaks[0]:peaks[1]])
# Do this on the width of the high amplitude area.
# Signals at 0 amplitude arbitrarily increase the frequency value

for i in range(0, 48):
    print(i)
    start_point = -1100 + 960*i
    end_point = -620 + 960*i
    frequency_samples = []
    for i in range(peaks[0] + start_point, peaks[0] + end_point):
        if data[i] > 0 and data[i - 1] < 0:
            frequency_samples.append(i)
        elif data[i] < 0 and data[i - 1] > 0:
            frequency_samples.append(i)

    # Values are really good between 10.8khz and 12.4khz for non-dithered

    consecutive_values = np.absolute(np.array(frequency_samples[1:]) - np.array(frequency_samples[:-1]))
    # print(consecutive_values)

    consecutive_values = [i if i < 4 else 1 for i in consecutive_values]

    # print(consecutive_values)

    frequency_mean = np.mean(consecutive_values, 0)
    # print(frequency_mean)

    frequency_1 = 48000/(2*frequency_mean)

    print(frequency_1)

    frequency = freq_from_fft(data[peaks[0] + start_point:peaks[0] + end_point], 48000)

    print(frequency)

    frequency = freq_from_autocorr(data[peaks[0] + start_point:peaks[0] + end_point], 48000)

    print(frequency)

    length, _ = find_peaks(data[peaks[0] + start_point:peaks[0] + end_point], None, 1, 1)
    print(len(length))

    print((len(length) * 100 + frequency_1)/2)

    print('-------------')

# Time for Plots
time = np.arange(1, len(data) + 1)
time_first_teager = np.arange(1, len(teager_first_data) + 1)
time_second_teager = np.arange(1, len(teager_second_data) + 1)
time_post_processed = np.arange(1, len(post_processed) + 1)
time_between_peaks = np.arange(1, 480 + 1)

# Begin Plotting
gs = gridspec.GridSpec(5,1)
fig = plt.figure()

# First Plot
ax = fig.add_subplot(gs[0])
ax.plot(time, data)
ax.set_ylabel(r'Raw Data', size = 16)
ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    labelbottom=False) # labels along the bottom edge are off

# Second Plot
ax = fig.add_subplot(gs[1])
ax.plot(time_first_teager, teager_first_data)
ax.set_ylabel(r'Teager 1', size = 16)
ax.time_between_peaks = np.arange(1, len(data[peaks[0]:peaks[2]]))
ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    labelbottom=False) # labels along the bottom edge are off

# Third Plot
ax = fig.add_subplot(gs[2])
ax.plot(time_second_teager, teager_second_data)
ax.set_ylabel(r'Teager 2', size = 16)
ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    labelbottom=False) # labels along the bottom edge are off

# Fourth Plot
ax = fig.add_subplot(gs[3])
ax.plot(time_post_processed, post_processed_data)
ax.set_ylabel(r'Peaks', size = 16)
ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    labelbottom=False) # labels along the bottom edge are off

# Fifth Plot
ax = fig.add_subplot(gs[4])
ax.plot(time_between_peaks, data[peaks[0] - 1100 + 960*11:peaks[0] - 620 + 960*11])
ax.set_ylabel(r'Peaks', size = 16)
ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    labelbottom=False) # labels along the bottom edge are off

plt.show()