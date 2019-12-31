from scipy.io import wavfile
from scipy.signal import find_peaks
from teager_py import Teager
import numpy as np
import time

def analyze(str directory):
    start_time = time.time()

    data = wavfile.read(directory)[1]

    data = [np.int64(i) for i in data]

    teager_first_data = Teager(np.abs(data), 'horizontal', 1)
    teager_first_data = np.array([np.int64(i) for i in teager_first_data])

    max_value = 0.01*np.amax(teager_first_data)

    teager_first_data[teager_first_data < max_value] = 0

    max_value = 5*max_value

    peaks = find_peaks(teager_first_data, None, 0, 1, max_value, [1.25, 2])[0]

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