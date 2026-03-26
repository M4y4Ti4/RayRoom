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
from rayroom.core.constants import FREQ_BANDS

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
    rirs, all_paths  = tracer.render(n_rays=20000,
            max_hops=150,
            rir_duration=2.0,
            record_paths=True,
            interference=True,
            ism_order=5,         # Enable Hybrid Mode
            show_path_plot=False, 
            parallel = False)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR
    plot_rir_per_band(rir_array = rir_array, rir_bands = rir_bands, fs = 44100, freq_bands = FREQ_BANDS)
    #extracting and sorting histogram by time
    hist = tracer.last_histogram[receiver1.name]
    hist_sorted = sorted(hist, key=lambda x: x[0])

    rir_raw = rir_array.sum(axis=1)
    direct = hist_sorted[0]
    time_direct, amp_direct, is_ism_direct, az_direct, el_direct = direct
    print(amp_direct)
    amp_direct_scalar = np.mean(np.abs(np.array(amp_direct))) #taking the mean of the band amplitudes for broadband scaling 

    first_reflection = hist_sorted[1]
    time_first, amp_first, is_ism_first, az_first, el_direct = first_reflection
    amp_first = np.mean(np.abs(np.array(amp_first)))

    print(amp_direct_scalar)

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

    # Check peak in each band separately
    freq_bands = [63, 125, 250, 500, 1000, 2000, 4000]
    for b, fc in enumerate(freq_bands):
        band = rir_array[:, b]
        peak_idx = np.argmax(np.abs(band))
        peak_time = peak_idx / fs * 1000
        peak_amp = np.max(np.abs(band))
        print(f"{fc}Hz: peak at {peak_time:.2f}ms, amp={peak_amp:.6f}")

    # Find the highest amplitude ISM reflection
    hist_ism = [(t,a,f,az,el) for t,a,f,az,el in hist if f]
    hist_ism_sorted = sorted(hist_ism, key=lambda x: np.mean(np.abs(np.array(x[1]))), reverse=True)

    print("Top 5 highest amplitude ISM reflections:")
    for t,a,f,az,el in hist_ism_sorted[:5]:
        amp = np.mean(np.abs(np.array(a)))
        print(f"  t={t*1000:.2f}ms  amp={amp:.6f}  az={az:.1f}°  el={el:.1f}°")


    np.savez(r"C:\Masters\Hybrid\RayroomProject\examples\initial_testing\rir_shoebox_cal_newscale.npz", 
             rir_total = rir_total, 
             rir_bands = rir_bands,
             fs = fs,
             brir_l = brir_l,
             brir_r = brir_r, 
             t_d = time_direct,
             t_r = time_first,
             amp_direct = amp_direct,
             amp_first = amp_first)
    print("saved")


if __name__ == "__main__":
    main()
