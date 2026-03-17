import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy 
from ..core.utils import smooth_tf

def plot_transfer_function(rir_total, fs, ax = None, label = "GA"):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(12, 5))

    n = len(rir_total)

    H = np.fft.rfft(rir_total, n=n)
    freqs = np.fft.rfftfreq(n, d=1/fs)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)
    smoothed = smooth_tf(freqs, magnitude_db, fraction=12)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, magnitude_db)
    ax.plot(freqs, smoothed, color = 'orange', label='1/3 octave smoothed')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Transfer Function")
    ax.set_xlim([0, 4000])
    ax.set_ylim([-80, 20])
    ax.grid(True, which='both', alpha=0.3)
    #ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000])
    #ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: str(int(x))))
    plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\Transfer function.png")
    plt.show()
    #plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\Transfer function.png")
    #plt.show()
    return ax



def overlay_DG(rir_total, fs):

    path = r"C:\Masters\DGBABY\edg-acoustics\examples\shoebox\output\TR_corrected_lc1_200Hz_2slc__10.npz"
    data_DG = np.load(path, allow_pickle=True)
    TR = data_DG["TR"]
    freqs = data_DG["freqs"]
    pos_idx = freqs >= 0

    # Compute magnitudes
    mag_DG = 20 * np.log10(np.abs(TR[pos_idx]) + 1e-12)
    freqs_pos = freqs[pos_idx]

    n = len(rir_total)
    H = np.fft.rfft(rir_total, n=n)
    freqs_GA = np.fft.rfftfreq(n, d=1/fs)
    mag_GA = 20 * np.log10(np.abs(H) + 1e-12)

    # Align levels at 80 Hz
    # Align using mean level between 40-120 Hz instead of a single point
    align_low, align_high = 160, 161
    mask_DG = (freqs_pos >= align_low) & (freqs_pos <= align_high)
    mask_GA = (freqs_GA >= align_low) & (freqs_GA <= align_high)
    offset = np.mean(mag_DG[mask_DG]) - np.mean(mag_GA[mask_GA])
    mag_GA_aligned = mag_GA + offset

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_pos, mag_DG, label="DG")
    ax.plot(freqs_GA, mag_GA_aligned, label=f"GA (aligned +{offset:.1f} dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Transfer Function")
    ax.set_xlim(0, 200)
    ax.set_ylim(-60, 10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.show()

def plot_rir(rir_total, fs):
    t = np.linspace(0, 2.0, len(rir_total))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(t, rir_total)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Impulse Response")
    ax.grid(True, which='both', alpha=0.3)
    plt.savefig(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\RIR.png")
    plt.show()

def plot_rir_per_band(rir_array, rir_bands, fs, freq_bands=None):
    if freq_bands is None:
        freq_bands = [63, 125, 250, 500, 1000, 2000, 4000]
    
    n_bands = len(freq_bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 2.5 * n_bands), sharex=True)
    t = np.arange(rir_array.shape[0]) / fs

    for i, freq in enumerate(freq_bands):
        axes[i].plot(t, rir_bands[i], alpha=0.8)
        axes[i].set_ylabel(f"{freq} Hz")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-np.max(np.abs(rir_bands[i])) * 1.2 - 1e-12,
                          np.max(np.abs(rir_bands[i])) * 1.2 + 1e-12)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("RIR per Frequency Band")
    plt.tight_layout()
    plt.show()
