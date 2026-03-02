import numpy as np


import soundfile as sf

data, samplerate = sf.read("rir_total.wav")
print("Shape:", data.shape)
print("Sample rate:", samplerate)
print("Duration (s):", len(data)/samplerate)