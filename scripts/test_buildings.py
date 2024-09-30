import os
import sys
import pathlib
import time
from functools import partial
import numpy as np

# make sure imports work by adding the aiml_virtual directory to path:
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(file_dir, '..'))

import aiml_virtual.scene as scene
import aiml_virtual.simulator as simulator
from aiml_virtual.mocap import dummy_mocap_source

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "scene_base.xml"))
    dummy_mocap_dict = {
        "airport": (np.array([-1, 1, 0]), np.array([0, 0, 0, 1])),
        "parking_lot_0": (np.array([0, 1, 0]), np.array([0, 0, 0, 1])),
        "landing_zone_10": (np.array([1, 1, 0]), np.array([0, 0, 0, 1])),
        "obst1": (np.array([-1, 0, 0]), np.array([0, 0, 0, 1])),
        "obst2": (np.array([0, 0, 0]), np.array([0, 0, 0, 1])),
        "payload 12": (np.array([1, 0, 0]), np.array([0, 0, 0, 1])),
        "bu1": (np.array([-1, -1, 0]), np.array([0, 0, 0, 1])),
        "bu2": (np.array([0, -1, 0]), np.array([0, 0, 0, 1])),
        "bu3": (np.array([1, -1, 0]), np.array([0, 0, 0, 1])),
    }
    frame_generator = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_dict, T=5)
    original_mocap = dummy_mocap_source.DummyMocapSource(frame_generator=frame_generator, fps=120)
    static_mocap = dummy_mocap_source.DummyMocapSource.freeze(original_mocap)
    time.sleep(0.1)  # leave time for the mocap to get its first frame
    scene.add_mocap_objects(static_mocap)
    sim = simulator.Simulator(scene, update_freq=500, target_fps=100)
    with sim.launch_viewer():
        while sim.viewer.is_running():
            sim.tick()
