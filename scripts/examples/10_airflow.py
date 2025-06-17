"""
This script
"""

import os
import sys
import pathlib
import numpy as np

# The lines under here are intended to make sure imports work, by adding parent folders to the path (i.e. the list
# of folders where the interpreter will look for a given package when you try to import it). This is to account for
# differences in what the interpreter identifies as your current working directory when launching these scripts
# from the command line as regular scripts vs with the -m option vs from PyCharm, as well as the script being placed
# in any depth of sub-sub-subfolder.
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual
xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator
from aiml_virtual.trajectory import dummy_drone_trajectory, skyc_trajectory
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object import bicycle
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import crazyflie, bumblebee, hooked_bumblebee

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_10.xml")
    bb = hooked_bumblebee.HookedBumblebee1DOF()
    bb.trajectory = dummy_drone_trajectory.DummyDroneTrajectory(np.array([-1, 0, 0.7]))
    scn.add_object(bb, "-1 0 0.7", "1 0 0 0", "0.5 0.5 0.5 1")

    scn.add_object(dynamic_object.BoxPayload(), "-0.9 0 1")

    sim = simulator.Simulator(scn)
    with sim.launch(speed=0.1):
        while not sim.display_should_close():
            sim.tick()  # tick steps the simulator, including all its subprocesses
            print(bb.prop_vel)



