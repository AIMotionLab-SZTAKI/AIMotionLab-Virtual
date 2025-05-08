import os
import sys
import pathlib
from xml.etree import ElementTree as ET

import numpy as np
from skyc_utils.skyc_maker import write_skyc, XYZYaw, Trajectory, TrajectoryType, LightProgram, Color
from skyc_utils.skyc_inspector import get_traj_data
import socket
import threading
from typing import Optional, Union
import copy
from scipy import interpolate
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

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
from aiml_virtual.scene import Scene
from aiml_virtual.simulator import Simulator
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory, extract_trajectories
from aiml_virtual.mocap.optitrack_mocap_source import OptitrackMocapSource


if __name__ == "__main__":
    scn = Scene(os.path.join(xml_directory, "demo_base.xml"))
    mocap = OptitrackMocapSource()
    scn.add_mocap_objects(mocap)
    sim = Simulator(scn)
    with sim.launch():
        while not sim.display_should_close():
            sim.tick()