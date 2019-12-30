from scipy.io import wavfile
from scipy.signal import find_peaks
from teager_py import Teager
import numpy as np
import time

cdef int func():
    start_time = time.time()

    cdef np.ndarray[np.int64_t, ndim=1] data = wavfile.read('Signals/start_signal_2.wav')[1]

    data = [np.int64(i) for i in data]

    cdef np.ndarray[np.int64_t, ndim=1] teager_first_data = np.empty(N, dtype=np.int64)
    teager_first_data = Teager(np.abs(data), 'horizontal', 1)

    cdef int max_value = np.amax(teager_first_data)

    cdef np.ndarray[np.int64_t, ndim=1] post_processed = [i if i > 0.01*max_value else 0 for i in teager_first_data]

    peaks = find_peaks(post_processed, None, 0, 1, 1/20*max_value, [1.25, 2])[0]

    largest_count = 0
    count = 0
    for i in range(len(peaks)):
        if (peaks[i] - peaks[i-1] < 1000):
            count += peaks[i] - peaks[i-1]
        else:
            count = 0
        largest_count = np.amax([largest_count, count])

    print("--- %s seconds ---" % (time.time() - start_time))

    return largest_count

func()