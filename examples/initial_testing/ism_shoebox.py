import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band 

from rayroom.engines.ism import ImageSourceRenderer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("thick_carpet"),
        "ceiling": get_material("drywall"),
        "front": get_material("brick"),
        "back": get_material("brick"),
        "left": get_material("brick"),
        "right": get_material("brick")
    }

    room = Room.create_shoebox([5, 4, 3], materials=mats)

    # 2. Add Objects
    # Source at (1, 1, 1.5)
    source    = Source("Speaker", [3.04, 2.59, 1.62], power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver1 = Receiver("persona", [4.26, 1.76, 1.62], radius=0.2)
    room.add_receiver(receiver1)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    # 3. Run Simulation
    tracer = ImageSourceRenderer(room)

    #setting source to a delta function: 
    fs = 44100
    impulse_length = 128   # 128 samples = ~2.9 ms
    delta_impulse = np.zeros(impulse_length)
    delta_impulse[0] = 1.0  # first sample is 1
    tracer.set_source_audio(source, delta_impulse)

    print("Starting simulation...")
    rirs = tracer.render(
            rir_duration=2.0,
            interference=False,
            ism_order=5)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR
    print(rir_bands)
    print(f"rir_total max freq content: {np.argmax(np.abs(np.fft.rfft(rir_total)))}")
    print(f"rir_array shape: {rir_array.shape}")
    print(f"rir_total shape: {rir_total.shape}")
    print("Total ISM paths:", len(receiver1.ism_paths))
    orders = [p["order"] for p in receiver1.ism_paths]
    print(np.bincount(orders))
    plot_rir_per_band(rir_array, rir_bands, fs=44100)
    plot_rir(rir_total, fs = 44100)
    plot_transfer_function(rir_total, fs = 44100)
    plot_transfer_function(rir_array[:,3], fs)
    np.savez(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data_ism.npz", rir_total = rir_total, fs = fs )
    print("saved")
    # Compare raw band vs filtered band TF
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Raw band 3 (500Hz column from rir_array)
    raw = rir_array[:, 3]
    H_raw = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(len(raw), 1/fs)
    axes[0].plot(freqs, 20*np.log10(np.abs(H_raw)+1e-12))
    axes[0].set_title("Raw 500Hz band (no additional filtering)")
    axes[0].set_xlim(0, 4000)
    axes[0].set_ylim(-60, 10)
    axes[0].grid(True, alpha=0.3)

    # Filtered band 3
    filt = rir_bands[3]
    H_filt = np.fft.rfft(filt)
    axes[1].plot(freqs, 20*np.log10(np.abs(H_filt)+1e-12))
    axes[1].set_title("Filtered 500Hz band (after bandpass)")
    axes[1].set_xlim(0, 4000)
    axes[1].set_ylim(-60, 10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()