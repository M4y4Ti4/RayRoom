import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band
from rayroom.room.visualize import plot_reverberation_time, debug_schroeder_curves

data = np.load(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data.npz")
rir_total = data["rir_total"]
fs = int(data["fs"])  # or data["fs"].item()

plot_rir(rir_total, fs)

plot_transfer_function(rir_total, fs)

overlay_DG(rir_total, fs)


