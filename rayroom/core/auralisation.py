"""code to perform HRTF convolution with RIR generated from ISM and raytracing hybrid"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.signal import fftconvolve
import sofar 


def get_hrir(hrtf, az_target=0, el_target=0):
    positions = hrtf.SourcePosition
    az = positions[:, 0]
    el = positions[:, 1]
    distances = np.sqrt((az - az_target)**2 + (el - el_target)**2)
    idx = np.argmin(distances)
    return hrtf.Data_IR[idx]  # shape (2, samples)


def load_hrtf(hrtf_path, fs_target):
    hrtf = sofar.read_sofa(hrtf_path)
    fs_hrtf = int(hrtf.Data_SamplingRate.flatten()[0])
    if fs_hrtf != fs_target: 
        from math import gcd 
        g = gcd(fs_target, fs_hrtf)
        hrtf.Data_IR = signal.resample_poly(hrtf.Data_IR, fs_target//g, fs_hrtf//g, axis = -1)
    return hrtf

def render_brir(histogram, src_xyz, rec_xyz, hrtf, fs=44100, duration=2.0, 
                freq_bands=None, interference=True):

    if freq_bands is None:
        freq_bands = [63, 125, 250, 500, 1000, 2000, 4000]

    n_bands   = len(freq_bands)
    src_xyz   = np.array(src_xyz)
    rec_xyz   = np.array(rec_xyz)
    n_samples = int(fs * duration)
    hrir_len  = hrtf.Data_IR.shape[-1]
    brir_len  = n_samples + hrir_len


    # Per-band binaural RIRs — complex to preserve interference
    brir_bands_l = np.zeros((brir_len, n_bands), dtype=complex)
    brir_bands_r = np.zeros((brir_len, n_bands), dtype=complex)

    histogram = sorted(histogram, key=lambda x: x[0])

    for entry in histogram:
        time, amp, is_ism, az, el = entry

        if time >= duration:
            continue

        # HRIR lookup — same for all bands (direction doesn't change with frequency)
        hrir   = get_hrir(hrtf, az_target=float(az), el_target=float(el))
        hrir_l = hrir[0].astype(complex)  # shape (hrir_len,)
        hrir_r = hrir[1].astype(complex)
        delay_left = np.where(np.abs(hrir_l) > 0.01*np.max(np.abs(hrir_l)))[0][0]
        delay_right = np.where(np.abs(hrir_r) > 0.01*np.max(np.abs(hrir_r)))[0][0]
        onset_delay = min(delay_left, delay_right)
        hrir_l = hrir_l[onset_delay:]
        hrir_r = hrir_r[onset_delay:]
        hrir_len = len(hrir_l)
        # Per-band amplitude
        amp_array = np.array(amp)  # shape (n_bands,)

        if is_ism and interference:
            # Keep complex amplitude — preserves phase interference
            scalars = amp_array  # complex, shape (n_bands,)
        elif is_ism and not interference:
            scalars = np.real(amp_array)
        else:
            # Ray tracing — energy to amplitude, random sign
            scalars = np.sqrt(np.abs(np.real(amp_array))) * np.random.choice([-1, 1])

        # Place each band at correct sample index
        idx = int(time * fs)
        if idx + hrir_len <= brir_len:
            for b in range(n_bands):
                brir_bands_l[idx:idx + hrir_len, b] += scalars[b] * hrir_l
                brir_bands_r[idx:idx + hrir_len, b] += scalars[b] * hrir_r

    # Take real part of each band
    brir_bands_l = np.real(brir_bands_l)  # (brir_len, n_bands)
    brir_bands_r = np.real(brir_bands_r)

    # Bandpass filter and sum each band — same as sum_frequency_bands
    from ..core.utils import sum_frequency_bands
    brir_l, _ = sum_frequency_bands(brir_bands_l, fs = 44100)
    brir_r, _ = sum_frequency_bands(brir_bands_r, fs = 44100)

    return brir_l[:n_samples], brir_r[:n_samples], brir_bands_l, brir_bands_r

def plot_brir(brir_l, brir_r, brir_bands_l, brir_bands_r, fs=44100, 
              freq_bands=None, duration=2.0):
    if freq_bands is None:
        freq_bands = [63, 125, 250, 500, 1000, 2000, 4000]

    n_plot = int(duration * fs)
    t = np.arange(n_plot) / fs * 1000  # ms
    print(f"brir_l length: {len(brir_l)}")
    print(f"n_plot: {n_plot}")
    print(f"t range: {t[0]:.2f}ms to {t[-1]:.2f}ms")

    # ---- Plot 1: Broadband BRIR L and R ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, brir_l[:n_plot], color='blue', linewidth=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("BRIR — Left Channel")
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(t, brir_r[:n_plot], color='red', linewidth=0.7)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("BRIR — Right Channel")
    axes[1].set_xlabel("Time (ms)")
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.suptitle("Broadband BRIR")
    plt.tight_layout()
    plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\output\brir_broadband.png")
    plt.show()

    # ---- Plot 2: Per-band BRIR (left only) ----
    n_bands = len(freq_bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 2 * n_bands), sharex=True)

    for b, fc in enumerate(freq_bands):
        axes[b].plot(t, brir_bands_l[:n_plot, b], color='blue', linewidth=0.7, label='L')
        axes[b].plot(t, brir_bands_r[:n_plot, b], color='red',  linewidth=0.7, label='R', alpha=0.7)
        axes[b].set_ylabel(f"{fc}Hz")
        axes[b].grid(True, linestyle='--', alpha=0.4)
        if b == 0:
            axes[b].legend(loc='upper right')

    axes[-1].set_xlabel("Time (ms)")
    plt.suptitle("Per-band BRIR (L=blue, R=red)")
    plt.tight_layout()
    plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\output\brir_per_band.png")
    plt.show()

    # ---- Plot 3: L vs R overlay broadband ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, brir_l[:n_plot], color='blue', linewidth=0.7, label='Left',  alpha=0.8)
    ax.plot(t, brir_r[:n_plot], color='red',  linewidth=0.7, label='Right', alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("BRIR — Left vs Right")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\output\brir_lr_overlay.png")
    plt.show()