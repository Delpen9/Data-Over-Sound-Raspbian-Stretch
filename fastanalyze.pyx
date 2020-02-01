from __future__ import division
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
import sys
from math import *
from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import time

## To build:
## python3 setup.py build_ext --inplace

def horizontal_teager(teager_array, teager_spread: int, teager_array_dimension: str):
    if (teager_array_dimension == '1D' or teager_array_dimension == '1d'): 
        i = teager_array[teager_spread:-teager_spread] * teager_array[teager_spread:-teager_spread]
        j = teager_array[(teager_spread - 1):(-teager_spread - 1)] * teager_array[(teager_spread + 1):None if (-teager_spread + 1) == 0 else (-teager_spread + 1)]
        return i - j

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
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation
    
    Pros: Best method for finding the true fundamental of any repeating wave, 
    even with strong harmonics or completely missing fundamental
    
    Cons: Not as accurate, currently has trouble with finding the true peak
    
    """
    # Calculate circular autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]
    
    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    # Also could zero-pad before doing circular autocorrelation.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    
    return fs / px

def analyze(str directory):
    start_time = time.time()

    data = wavfile.read(directory)[1]

    data = [np.int64(i) for i in data]

    teager_first_data = horizontal_teager(np.abs(data), 1, '1D')
    teager_second_data = horizontal_teager(np.abs(teager_first_data), 1, '1D')
    
    max_value = np.amax(teager_second_data)*0.01
    post_processed = [i if i > max_value else 0 for i in teager_second_data]

    max_value = 5*max_value
    peaks = find_peaks(post_processed, None, 0, 1, max_value, [1.25, 2])[0]

    chunk_list = []
    post_processed_data_chunk = []
    for i in range(len(peaks)):
        if len(chunk_list) == 0:
            chunk_list.append(peaks[i])
        elif peaks[i] - peaks[i - 1] < 200:
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

    post_processed_data = [1 for i in teager_second_data]
    for i in range(len(teager_second_data)):
        if i not in post_processed_data_chunk:
            post_processed_data[i] = 0

    peaks, _ = find_peaks(post_processed_data, None, 0, 1, 1)

    freq1 = freq_from_autocorr(data[peaks[0]:peaks[0] + 48000], 48000)
    freq2 = freq_from_autocorr(data[peaks[len(peaks) - 1] - 48000:peaks[len(peaks) - 1]], 48000)
    count = peaks[len(peaks) - 1] - peaks[0]
    print(count)
    print(freq1)
    print(freq2)
    print("--- %s seconds ---" % (time.time() - start_time))

    return not(9000 <= freq1 <= 10000 or 9000 <= freq2 <= 10000)

## for i in range(50):
    ## val = analyze('/media/ian/Elements/Data/1_100th_second/1_10th_second/wakeupsignal_10000_11000_' + str(i*20) + '.wav')
    ## val = analyze('/media/ian/Elements/Data/wakeupsignal_11000_12000_' + str(i*20) + '.wav')
    ## val = analyze('/media/ian/Elements/Data/1_10th_second/wakeupsignal_5000_10000_1_10th_second_' + str(i*20) + '.wav')
    ## print('Around 48000:', val)
    ## print(str(20*i))
