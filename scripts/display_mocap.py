"""
For Mr. Szécsi
workflow:
In order to visualize, a Simulator is needed (stepping this simulator is what renders the image). The simulator needs
a Scene to simulate (Scenes define the objects that will be involved in a simulation).
To add a mocap drone to the scene, this mocap drone will need a source for its mocap data: this functionality is
provided by MocapSources, such as a DummyMocapSource.
DummyMocapSources need a function that:
    -takes no arguments
    -generates frames, which are dictionaries with string keys and tuple[np.ndarray, np.ndarray] values (pos, quat)

Workflow:
make scene (needs a base, such as the model of the demo area, or an empty area)
make frame generator (functools.partial may be needed)
make DummyMocapSource based on frame generator
make MocapCrazyflie based on mocap source
add MocapCrazyflie to scene
make simulator based on scene
run simulation loop infinitely

Controls in the viewer are mostly mouse clicks/wheel, and spacebar.
Tab and shift+Tab toggle left and right side menus, but you don't need them.
Video capture isn't implemented yet but you can do that using regular PC screen capture.

Haven fun!
"""
import os
import sys
import pathlib
import numpy as np
from functools import partial
# make sure imports work by adding the aiml_virtual directory to path:
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(file_dir, '..'))

import aiml_virtual.scene as scene
import aiml_virtual.simulator as simulator
import aiml_virtual.simulated_object.mocap_object.mocap_drone.mocap_crazyflie as mcf
from aiml_virtual.mocap import dummy_mocap_source


if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"))
    dummy_mocap_start_poses = {
        "cf0": (np.array([0, 0, 0.5]), np.array([0, 0, 0, 1]))
    }
    frame_generator = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_start_poses, T=5)
    mocap = dummy_mocap_source.DummyMocapSource(frame_generator=frame_generator, fps=120)
    cf = mcf.MocapCrazyflie(mocap, "cf0")
    scene.add_object(cf)
    sim = simulator.Simulator(scene)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()