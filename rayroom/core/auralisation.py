"""code to perform HRTF convolution with RIR generated from ISM and raytracing hybrid"""
import slab
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav

hrtf = slab.HRTF.kemar()
print(hrtf)