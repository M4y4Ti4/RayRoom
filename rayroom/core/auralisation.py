"""code to perform HRTF convolution with RIR generated from ISM and raytracing hybrid"""
import slab
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.signal import fftconvolve
import sofar 

data = np.load(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_data.npz")
hrtf = sofar.read_sofa(r"C:\Masters\HRTF\KEMAR_GRAS_EarSim_LargeEars_FreeFieldComp_44kHz.sofa")
sample_rate, audio_data = wav.read(r"C:\Masters\audio\voz.wav")
audio_resampled = signal.resample_poly(audio_data, 44100, 8000)
fs_target = 44100
fs_hrtf = int(hrtf.Data_SamplingRate.flatten()[0])

# Load RIR data
rir_total = data["rir_total"]
az_array = data["az"]
el_array = data["el"]

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
brir_left = data["brir_left"]
brir_right = data["brir_right"]

# --- Plot BRIRs ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

t = np.arange(len(rir_total)) / fs_target
axes[0].plot(t, rir_total)
axes[0].set_title("Room RIR")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

t_brir = np.arange(len(brir_left)) / fs_target
axes[1].plot(t_brir, brir_left,  label='Left',  alpha=0.8)
axes[1].plot(t_brir, brir_right, label='Right', alpha=0.8)
axes[1].set_title("BRIR (Left and Right)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# TF of BRIRs
H_left  = np.fft.rfft(brir_left)
H_right = np.fft.rfft(brir_right)
freqs   = np.fft.rfftfreq(len(brir_left), 1/fs_target)
axes[2].plot(freqs, 20*np.log10(np.abs(H_left)  + 1e-12), label='Left',  alpha=0.8)
axes[2].plot(freqs, 20*np.log10(np.abs(H_right) + 1e-12), label='Right', alpha=0.8)
axes[2].set_title("BRIR Transfer Function")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude (dB)")
axes[2].set_xlim(0, 4000)
axes[2].set_ylim(-60, 10)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

audio_left = signal.fftconvolve(brir_left, audio_resampled, mode = 'full')
audio_right = signal.fftconvolve(brir_right, audio_resampled, mode='full')

max_len = max(len(audio_left), len(audio_right))
audio_left = np.pad(audio_left, (0, max_len - len(audio_left)))
audio_right = np.pad(audio_right, (0, max_len - len(audio_right)))

max_val = np.max([np.max(np.abs(audio_left)), np.max(np.abs(audio_right))])
audio_left  /= max_val
audio_right /= max_val

stereo = np.column_stack((audio_left, audio_right)).astype(np.float32)
wav.write(r"C:\Masters\audio\stereo_output.wav", fs_target, stereo)
"""

# --- Diagnostic plots ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot RIR
t_rir = np.arange(len(rir_total)) / fs_target
axes[0].plot(t_rir, rir_total)
axes[0].set_title(f"Room RIR (max={np.max(np.abs(rir_total)):.4f})")
axes[0].set_xlabel("Time (s)")
axes[0].grid(True, alpha=0.3)

# Plot HRIR
t_hrir = np.arange(len(hrir_left)) / fs_target * 1000  # ms
axes[1].plot(t_hrir, hrir_left, label='Left')
axes[1].plot(t_hrir, hrir_right, label='Right')
axes[1].set_title(f"HRIR (max={np.max(np.abs(hrir_left)):.4f})")
axes[1].set_xlabel("Time (ms)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot audio
t_audio = np.arange(len(audio_resampled)) / fs_target
axes[2].plot(t_audio, audio_resampled)
axes[2].set_title(f"Audio (max={np.max(np.abs(audio_resampled)):.4f})")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

brir_left = signal.fftconvolve(rir_total, hrir_left, mode = 'full')
brir_right = signal.fftconvolve(rir_total, hrir_right, mode = 'full')


audio_left = signal.fftconvolve(brir_left, audio_resampled, mode = 'full')
audio_right = signal.fftconvolve(brir_right, audio_resampled, mode = 'full')

max_len = max(len(audio_left), len(audio_right))
audio_left = np.pad(audio_left, (0, max_len - len(audio_left)))
audio_right = np.pad(audio_right, (0, max_len - len(audio_right)))

max_val = np.max([np.max(np.abs(audio_left)), np.max(np.abs(audio_right))])
audio_left  /= max_val
audio_right /= max_val

stereo = np.column_stack((audio_left, audio_right)).astype(np.float32)
wav.write(r"C:\Masters\audio\stereo_output.wav", fs_target, stereo)





sample_rate, audio_data = wav.read(r"C:\Masters\audio\voz.wav")

audio_resampled = signal.resample_poly(audio_data, 44100, 8000)

output = signal.fftconvolve(audio_resampled, rir_total, mode = 'full')

output = output / np.max(np.abs(output))

wav.write(r"C:\Masters\audio\test_output_drywall.wav", 44100, output.astype(np.float32))

H = np.fft.rfft(output)
freqs = np.fft.rfftfreq(len(output), 1/44100)
plt.plot(freqs, 20*np.log10(np.abs(H)+1e-12))
plt.ylim(-50, 100)
plt.xlim(0, 4000)
plt.show()
"""