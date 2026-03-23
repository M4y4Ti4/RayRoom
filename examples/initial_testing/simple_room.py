import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band, plot_rir_components
from rayroom.room.visualize import plot_reverberation_time
import random
from rayroom.core.auralisation import load_hrtf, render_brir, plot_brir, get_hrir


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
    receiver1 = Receiver("persona", [4.26, 1.76, 1.62], radius=0.09)
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
    rirs, all_paths  = tracer.render(n_rays=2000,
            max_hops=150,
            rir_duration=2.0,
            record_paths=True,
            interference=True,
            ism_order=5,         # Enable Hybrid Mode
            show_path_plot=True, 
            parallel = False)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR

    hist = tracer.last_histogram[receiver1.name]

    rir_ism, rir_ray, rir_hybrid = plot_rir_components(hist, fs = fs)
    hrtf = load_hrtf(r"C:\Masters\HRTF\KEMAR_GRAS_EarSim_LargeEars_FreeFieldCompMinPhase_44kHz.sofa", fs_target=44100)

    brir_l, brir_r, brir_bands_l, brir_bands_r = render_brir(
    histogram=hist,
    src_xyz=[3.04, 2.59, 1.62],
    rec_xyz=[4.26, 1.76, 1.62],
    hrtf=hrtf,
    fs=fs,
    duration=2.0,
    interference=True)

    plot_brir(brir_l, brir_r, brir_bands_l, brir_bands_r, fs = 44100)

    np.savez(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data.npz", 
             rir_total = rir_total, 
             rir_array = rir_array,
             fs = fs,
             brir_l = brir_l,
             brir_r = brir_r)
    print("saved")


if __name__ == "__main__":
    main()
