import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import generate_rir
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer, AmbisonicReceiver
from scipy.io import wavfile
from scipy.signal import fftconvolve
from rayroom.core.constants import DEFAULT_SAMPLING_RATE


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "front": get_material("brick"),
        "back": get_material("brick"),
        "left": get_material("concrete"),
        "right": get_material("glass")
    }

    room = Room.create_shoebox([5, 4, 3], materials=mats)

    # 2. Add Objects
    # Source at (1, 1, 1.5)
    source = Source("Speaker", [1, 3, 1.5], power=1.0)
    room.add_source(source)

    # Receiver person facing in -ve y direction
    receiver = Receiver('receivepls', [4, 3, 1.5], radius = 0.2)
    room.add_receiver(receiver)
    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    #create delta impulse
    impulse = np.array([1.0])

    # 3. Run Simulation
    renderer = HybridRenderer(room, fs = DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)
    renderer.set_source_audio
    print("Starting simulation...")
    #tracer.generate_rir_only(source, n_rays=20000, max_hops=30)
    tracer.set_source_audio(source, impulse, gain=10.0)
    receiver_outputs, rirs = tracer.render()

    print(rirs)
  

"""
#if performing binaural aurlisation based on two receivers 
    #np.save('rir.npy', rir)
    rirl_normalised = rir_left/ (np.max(np.abs(rir_left)) + 1e-10)
    rirr_normalised = rir_right/ (np.max(np.abs(rir_right)) + 1e-10)

    wavfile.write('rir_left.wav', 44100, rirl_normalised.astype(np.float32))
    wavfile.write('rir_right.wav', 44100, rirr_normalised.astype(np.float32))

    fs, audio = wavfile.read(r"C:\\Masters\\RayroomProject\\rayroom\\examples\\engines\\audio_sources\\speaker_1.wav")
    left_channel = fftconvolve(audio, rir_left, mode = 'full')
    right_channel = fftconvolve(audio, rir_right, mode = 'full')

    binaural_audio = np.stack([left_channel, right_channel], axis = 1)

    binaural_audio /= np.max(np.abs(binaural_audio))

    wavfile.write('binaural_output.wav', fs, binaural_audio.astype(np.float32))
"""

if __name__ == "__main__":
    main()
