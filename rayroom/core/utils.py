import numpy as np

def generate_rir(histogram, fs=44100, duration=2.0, random_phase=True, collapse_bands = False):
    """Generates a Room Impulse Response (RIR) from a time-energy histogram.

    This function converts a list of reflection arrival times and their
    corresponding amplitudes into a discrete-time RIR signal. This is a
    common way to synthesize an RIR from the output of a ray tracing or
    image source model simulation.

    Responsibilities:
      * Convert a time-based energy histogram to a sampled RIR.
      * Handle the quantization of arrival times to sample indices.
      * Optionally apply random phase to simulate diffuse reflections.
      * Ensure the RIR has the specified duration and sample rate.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            # A simple histogram of reflections: (time_in_seconds, amplitude)
            reflection_histogram = [
                (0.01, 1.0),   # Direct sound
                (0.05, 0.5),   # First reflection
                (0.08, 0.3),   # Second reflection
                (0.08, 0.25)   # Another reflection at the same time
            ]
            
            fs = 44100
            rir_duration = 0.2  # seconds
            
            rir = rt.core.utils.generate_rir(
                reflection_histogram, fs=fs, duration=rir_duration
            )
            
            # The result is a NumPy array representing the RIR

    :param histogram: A list of tuples, where each tuple contains the arrival
                      time (in seconds) and amplitude of a reflection.
    :type histogram: list[tuple[float, float]]
    :param fs: The sampling frequency in Hertz (Hz). Defaults to 44100.
    :type fs: int, optional
    :param duration: The desired duration of the RIR in seconds. Defaults to 2.0.
    :type duration: float, optional
    :param random_phase: If `True`, applies a random sign flip to each
                         reflection to simulate phase variations from diffuse
                         surfaces. Defaults to `True`.
    :type random_phase: bool, optional
    :return: The generated Room Impulse Response.
    :rtype: np.ndarray
    """
    rir_len = int(fs*duration)

    if not histogram:
        return np.zeros((rir_len,0))

    # Sort by time
    histogram.sort(key=lambda x: x[0])

    if len(histogram[0]) == 3: 
        times = np.array([t for t, _, _ in histogram])
        raw_amps = [a for _, a, _ in histogram]
        is_ism = np.array([flag for _, _, flag in histogram])
    else: 
        times = np.array([t for t, _ in histogram])
        raw_amps = np.array([a for t, a in histogram])
        is_ism = np.zeros(len(raw_amps), dtype = bool)

    # Discard late reflections
    valid = times < duration
    times = times[valid]
    raw_amps = [raw_amps[i] for i in range(len(raw_amps)) if valid[i]]
    is_ism = is_ism[valid]

    if len(times) == 0:
        return np.zeros((rir_len, len(raw_amps[0]) if raw_amps else 0))
    
    n_bands = len(raw_amps[0])
    final_amps = np.zeros((len(raw_amps), n_bands))
    

    for i, (amp, ism) in enumerate(zip(raw_amps, is_ism)):
        if ism:
            final_amps[i] = np.real(amp)
        else:
            a = np.sqrt(np.real(amp))
            if random_phase:
                a *= np.random.choice([-1, 1])
            final_amps[i] = a

    rir = np.zeros((rir_len, n_bands))
    indices = (times * fs).astype(int)
    for b in range(n_bands):
        np.add.at(rir[:, b], indices, final_amps[:, b])
    print("\n--- First 5 histogram entries ---")
    for i, entry in enumerate(histogram[:5]):
        t, amp, is_ism_flag = entry if len(entry) == 3 else (*entry, "unknown")
        print(f"  t={t:.4f}s  is_ism={is_ism_flag}  amp={np.real(amp)[:2]}")
    if collapse_bands:
        return rir.sum(axis=1)
    return rir
