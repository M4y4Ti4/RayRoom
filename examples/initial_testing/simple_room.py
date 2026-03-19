import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band, plot_rir_components
from rayroom.room.visualize import plot_reverberation_time
import random


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("carpet"),
        "ceiling": get_material("drywall"),
        "front": get_material("drywall"),
        "back": get_material("drywall"),
        "left": get_material("drywall"),
        "right": get_material("drywall")
    }

    room = Room.create_shoebox([5, 4, 3], materials=mats)
    # 2. Add Objects
    # Source at (1, 1, 1.5)
    source    = Source("Speaker", [3.04, 2.59, 1.62], power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver1 = Receiver("persona", [4.26, 2.59, 1.62], radius=0.09)
    room.add_receiver(receiver1)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    # 3. Run Simulation
    tracer = HybridRenderer(room)

    #setting source to a delta function: 
    fs = 44100
    impulse_length = 128   # 128 samples = ~2.9 ms
    delta_impulse = np.zeros(impulse_length)
    delta_impulse[0] = 1.0  # first sample is 1
    tracer.set_source_audio(source, delta_impulse)

    print("Starting simulation...")
    #tracer.generate_rir_only(source, n_rays=20000, max_hops=30)
    rirs, all_paths, brir  = tracer.render(n_rays=20000,
            max_hops=150,
            rir_duration=2.0,
            record_paths=True,
            interference=True,
            ism_order=5,         # Enable Hybrid Mode
            show_path_plot=True, 
            parallel = False)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR
    brir_left, brir_right = brir[receiver1.name]

    directions = tracer.last_directions.get(receiver1.name, [])
    times_dir = np.array([d[0] for d in directions])
    azimuths = np.array([d[1] for d in directions])
    elevations = np.array([d[2] for d in directions])

    hist = tracer.last_histogram[receiver1.name]
    rir_ism, rir_ray, rir_hybrid = plot_rir_components(hist, fs = fs)

    np.savez(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data.npz", 
             rir_total = rir_total, 
             fs = fs,
             az = azimuths, 
             el = elevations, 
             brir_left = brir_left,
             brir_right = brir_right)
    print("saved")

    plot_rir_per_band(rir_array, rir_bands, fs=44100)
    plot_reverberation_time(rir_array, fs)
    raw_band = rir_array[:, 3]
    H = np.fft.rfft(raw_band)
    freqs = np.fft.rfftfreq(len(raw_band), 1/fs)
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    plt.plot(freqs, mag)
    plt.xlim(0, 4000)
    plt.show()

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    freq_bands = [63, 125, 250, 500, 1000, 2000, 4000]

    for i, freq in enumerate(freq_bands):
        # Raw band
        raw = rir_array[:, i]
        H_raw = np.fft.rfft(raw)
        freqs = np.fft.rfftfreq(len(raw), 1/fs)
        axes[i].plot(freqs, 20*np.log10(np.abs(H_raw)+1e-12), alpha=0.7, label='raw')
        
        # Filtered band
        filt = rir_bands[i]
        H_filt = np.fft.rfft(filt)
        axes[i].plot(freqs, 20*np.log10(np.abs(H_filt)+1e-12), alpha=0.7, label='filtered')
        
        axes[i].set_title(f"{freq} Hz band")
        axes[i].set_xlim(0, 4000)
        axes[i].set_ylim(-60, 10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Magnitude (dB)")

            # Hide the last empty subplot (7 bands, 8 subplots)
    axes[-1].set_visible(False)

    plt.suptitle("Transfer Function per Frequency Band (Raw vs Filtered)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
