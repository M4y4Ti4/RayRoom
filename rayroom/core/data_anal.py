import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_transfer_function(rir_total, fs):

    n = len(rir_total)

    H = np.fft.rfft(rir_total, n=n)
    freqs = np.fft.rfftfreq(n, d=1/fs)

    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, magnitude_db)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Transfer Function")
    ax.set_xlim([0, 200])
    ax.set_ylim([-60, 10])
    ax.grid(True, which='both', alpha=0.3)
    #ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000])
    #ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: str(int(x))))
    plt.show()


def plot_rir(rir_total, fs):
    t = np.linspace(0, 2.0, len(rir_total))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(t, rir_total)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Impulse Response")
    ax.grid(True, which='both', alpha=0.3)
    plt.show()

