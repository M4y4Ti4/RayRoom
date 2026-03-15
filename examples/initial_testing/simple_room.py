import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band
from rayroom.room.visualize import plot_reverberation_time
import random


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("carpet"),
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
    tracer = HybridRenderer(room)

    #setting source to a delta function: 
    fs = 44100
    impulse_length = 128   # 128 samples = ~2.9 ms
    delta_impulse = np.zeros(impulse_length)
    delta_impulse[0] = 1.0  # first sample is 1
    tracer.set_source_audio(source, delta_impulse)

    print("Starting simulation...")
    #tracer.generate_rir_only(source, n_rays=20000, max_hops=30)
    rirs, all_paths = tracer.render(n_rays=2000,
            max_hops=150,
            rir_duration=2.0,
            record_paths=True,
            interference=True,
            ism_order=5,         # Enable Hybrid Mode
            show_path_plot=True, 
            parallel = False)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR
    for i, band in enumerate(rir_bands):
        print(f"Band {[63,125,250,500,1000,2000,4000][i]}Hz: max={np.max(np.abs(band)):.6e}")
    print(rir_bands)
    print(f"rir_total max freq content: {np.argmax(np.abs(np.fft.rfft(rir_total)))}")
    print(f"rir_array shape: {rir_array.shape}")
    print(f"rir_total shape: {rir_total.shape}")
    # After simulation, count ISM vs ray contributions
    hist = receiver1.amplitude_histogram
    ism_count = sum(1 for entry in hist if entry[2] == True)  
    ray_count = sum(1 for entry in hist if entry[2] == False)
    print(f"ISM reflections: {ism_count}")
    print(f"Ray reflections: {ray_count}")

    # Also check energy balance
    ism_energy = sum(np.sum(np.abs(entry[1])**2) for entry in hist if entry[2] == True)
    ray_energy = sum(np.sum(np.abs(entry[1])**2) for entry in hist if entry[2] == False)
    print(f"ISM energy: {ism_energy:.6f}")
    print(f"Ray energy: {ray_energy:.6f}")
    print(f"Ray/ISM energy ratio: {ray_energy/ism_energy:.1f}x")
    np.savez(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data.npz", rir_total = rir_total, fs = fs )
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

if __name__ == "__main__":
    main()
