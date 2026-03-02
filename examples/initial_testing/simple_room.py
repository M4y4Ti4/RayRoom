import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer

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
    source = Source("Speaker", [1, 1, 1.5], power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver1 = Receiver("persona", [4, 3, 1.5], radius=0.2)
    room.add_receiver(receiver1)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    # 3. Run Simulation
    tracer = HybridRenderer(room)

    #setting source to a delta function: 
    fs = 44100
    impulse_length = 128 * fs  # 128 samples = ~2.9 ms
    delta_impulse = np.zeros(impulse_length)
    delta_impulse[0] = 1.0  # first sample is 1
    tracer.set_source_audio(source, delta_impulse)

    print("Starting simulation...")
    #tracer.generate_rir_only(source, n_rays=20000, max_hops=30)
    rirs, all_paths = tracer.render(n_rays=10000,
            max_hops=100,
            rir_duration=1.0,
            record_paths=True,
            interference=False,
            ism_order=2,         # Enable Hybrid Mode
            show_path_plot=True, 
            parallel = False)
    rir_array = rirs[receiver1.name]
    rir_total = rir_array.sum(axis=1)
    np.save("rir_total", rir_total)
    t = np.linspace(0, 1.0, len(rir_total))
    plt.figure()
    plt.plot(t, rir_total)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("RIR")
    plt.show()

    # Optional: scaled WAV for HRTF convolution
    import soundfile as sf
    rir_scaled = rir_total / np.max(np.abs(rir_total))
    print(len(rir_scaled))
    sf.write("rir_total.wav", rir_scaled.astype('float32'), samplerate=fs)

    times, energies = zip(*receiver1.amplitude_histogram)
    times = np.array(times)
    energies = np.array(energies)

    plt.figure()

    N_BANDS = energies.shape[1]
    colors = plt.cm.viridis(np.linspace(0,1,N_BANDS))

    for b in range(N_BANDS):
        plt.hist(times, bins = 50, weights=energies[:,b], alpha=0.2, color=colors[b], label=f'Band{b+1}')
    
    plt.show()

    """
    #access image sources
    image_sources = tracer.ism_engine.last_image_sources
    print(len(image_sources))


    #access ray_paths
    ray_paths = all_paths.get(source.name, [])
    print(f"Recorded {len(ray_paths)} ray paths")
    
    for path in receiver1.ism_paths:
        print(f"Order {path['order']}, time={path['time']:.4f}s, energy={path['energy']}")
        for p in path['points']:
            print(f"  {np.round(p, 3)}")
    
    print(f"valid paths for receiver1: {len(receiver1.ism_paths)}")

"""

if __name__ == "__main__":
    main()
