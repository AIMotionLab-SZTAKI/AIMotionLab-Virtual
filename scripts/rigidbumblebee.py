"""
This script shows how dynamic objects work.
"""

import os
import sys
import pathlib
import numpy as np

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")
project_root = project_root.resolve().as_posix()

from aiml_virtual import scene, simulator
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object import bicycle
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import bumblebee

if __name__ == "__main__":
    # As mentioned in 2_build_scene.py, we can simulate physics using DynamicObjects. So far we've only seen dynamic
    # objects that had no actuators. Let's change that, and build a scene with dynamic objects based on the empty
    # checkerboard scene base!
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_3.xml")

    bb = bumblebee.FixedPropBumblebee()
    scn.add_object(bb, "0 0 0.5", "1 0 0 0", "0.5 0.5 0.5 1")
    bb.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([0, 0, 0.5]))

    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses




