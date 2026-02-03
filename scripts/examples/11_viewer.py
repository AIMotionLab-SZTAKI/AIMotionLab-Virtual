"""
This script demonstrates the capability of the skyc viewer to play .skyc files.
"""
import os
import sys
import pathlib

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
from aiml_virtual.skyc_viewer import SkycViewer

if __name__ == "__main__":
    # At AIMotionLab we mostly upload skyc trajectories to drones. We may want to check out what they look like,
    # before we upload them to the drones. The SkycViewer class allows us to do just that.
    viewer = SkycViewer(os.path.join(aiml_virtual.skyc_folder, "three_drones_collisions.skyc"))
    # The skyc viewer has three main functions:
    # viewer.plot() # This function uses skyc_utils to plot the trajectories in matplotlib.
    # This function plays the trajectories using raw mocap data, meaning that the crazyflies are mocap objects, following
    # the trajectory exactly as it's defined.
    viewer.play_raw()
    # This function plays the trajectories using a controller, meaning that the crazyflies are dynamic objects,
    # and they try to follow the trajectory using a geometric controller.
    viewer.play_with_controller()