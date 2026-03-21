import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band 
from rayroom.engines.ism import ImageSourceRenderer
from rayroom.room.materials import Material
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

alpha = 0.2
uniform_mat = Material("uniform_abs", 
                       absorption = [alpha]*7, 
                       scattering=[0.0]*7)
def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": uniform_mat,
        "ceiling": uniform_mat,
        "front": uniform_mat,
        "back": uniform_mat,
        "left": uniform_mat,
        "right": uniform_mat
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
            ism_order=2)
    rir_array = rirs[receiver1.name]
    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ism_order10_highabs")

    np.savez(
        output_path,
        rir_total=rir_total,
        rir_bands=np.array(rir_bands),
        rir_array=rir_array,
        fs=np.array(44100)
    )
    print(f"Saved to: {output_path}.npz")    
if __name__ == "__main__":
    main()