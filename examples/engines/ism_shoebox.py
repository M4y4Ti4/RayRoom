import os
import sys
import argparse

from rayroom.engines.ism import ImageSourceRenderer
from rayroom.analytics.performance import PerformanceMonitor
from rayroom.room import objects, Room, get_material, Source, Receiver

from demo_utils import (
    generate_layouts,
    save_room_mesh,
    run_metrics_and_save,
    save_performance_metrics,
)
from rayroom.room.database import DemoRoom
from rayroom.core.constants import DEFAULT_SAMPLING_RATE

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
    source = Source("Speaker", [1, 1, 1.5], power=10.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver = Receiver("Mic", [4, 3, 1.5], radius=1)
    room.add_receiver(receiver)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    #perform image source method in this room
    print("Initializing ISM Renderer...")
    renderer = ImageSourceRenderer(room, fs=DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    # build the path to the audio files folder relative to this script file
    base_path = os.path.join(os.path.dirname(__file__), "audio_sources")

    if not os.path.exists(os.path.join(base_path, "speaker_1.wav")):
        print("Error: Example audio files not found.")
        exit(1)

    renderer.set_source_audio(source, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    print("Starting ISM Rendering pipeline...")

    ism_order = 8
    print(f"  - ISM Order: {ism_order}")

    with PerformanceMonitor() as monitor:
        outputs, rirs = renderer.render(
            ism_order=ism_order,
            rir_duration=1.5,
            interference=False  # Set to True for phase interference effects if desired
        )
    print(outputs, rirs)
main()