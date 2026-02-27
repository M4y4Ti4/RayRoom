"""
This module defines globally used physical and mathematical constants.

This central location ensures consistency across the simulation and analysis
tools in the `rayroom` library.

Constants:
  * ``C_SOUND``: The speed of sound in air at 20°C.
"""
C_SOUND = 343.0  # Speed of sound in air in m/s at 20°C

DEFAULT_SAMPLING_RATE = 44100

FREQ_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

N_BANDS = len(FREQ_BANDS)