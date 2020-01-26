import numpy as np
from scipy.io import wavfile
import pyaudio
import sox
import os

print("A new file was created.")

## 44100 is sometimes the best
## 88200 is the best
## 176400
## Notes: Raising the amplitude above dither may improve accuracy (square wave-ish)
sampleRate = 48000
frequency_list = [2000, 8000]
duration = 0.01
frequency = frequency_list[0]

t = np.linspace(0, duration, int(sampleRate * duration))
y = 0*np.sin(frequency * 2 * np.pi * t)

try:
	move_ahead = duration/(sampleRate * duration)
	for i in range(1, 300):
		if (i < 100 or i > 199):
			if (i % 2 == 0):
				t = np.linspace(duration + move_ahead, 2*duration, int(sampleRate * duration))
				y = np.append(y, 0.8*np.sin(frequency * 2 * np.pi * t))
			else:
				t = np.linspace(duration + move_ahead, 2*duration, int(sampleRate * duration))
				y = np.append(y, 0*np.sin(frequency * 2 * np.pi * t))
		else:
			frequency = frequency_list[1]
			t = np.linspace(duration + move_ahead, 2*duration, int(sampleRate * duration))
			y = np.append(y, 0*np.sin(frequency * 2 * np.pi * t))
			

	wavfile.write('Sine.wav', sampleRate, y)

	os.system('sox Sine.wav -e signed-integer -r 48000 -c 1 -b 16 Better_Sine.wav')
	
	pass

except (TypeError):
	pass
