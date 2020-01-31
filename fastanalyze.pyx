from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import time

## To build:
## python3 setup.py build_ext --inplace

def horizontal_teager(teager_array, teager_spread: int, teager_array_dimension$
    if (teager_array_dimension == '1D' or teager_array_dimension == '1d'):
        i = teager_array[teager_spread:-teager_spread] * teager_array[teager_s$
        j = teager_array[(teager_spread - 1):(-teager_spread - 1)] * teager_ar$
        return i - j

def analyze(str directory):
    start_time = time.time()

    data = wavfile.read(directory)[1]

    data = [np.int64(i) for i in data]

    teager_first_data = horizontal_teager(np.abs(data), 1, '1D')
    teager_second_data = horizontal_teager(np.abs(teager_first_data), 1, '1D')
    max_value = np.amax(teager_second_data)*0.01

    post_processed = [i if i > max_value else 0 for i in teager_second_data]

    peaks = find_peaks(teager_second_data, None, 0, 1, max_value, [1.25, 2])[0]

    largest_count = 0
    count = 0
    for i in range(len(peaks)):
        if (peaks[i] - peaks[i-1] < 1500):
            count += peaks[i] - peaks[i-1]
        else:
            count = 0
        largest_count = np.amax([largest_count, count])
    print(largest_count)
    print("--- %s seconds ---" % (time.time() - start_time))
    return not(56000 >= largest_count >= 40000)
