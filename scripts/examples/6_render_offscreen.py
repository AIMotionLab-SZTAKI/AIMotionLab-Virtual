"""
This script shows how you can render a simulation without a display
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
xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "example_scene_2.xml"), save_filename="example_scene_2.xml")
    # As noted in 5_record.py, you can have a display without rendering anything, like we did in the
    # first four examples, but you can also render without displaying anything.
    # For now, let's render a video from the second example's scene without actually displaying anything.
    sim = simulator.Simulator(scn)
    with sim.launch(with_display=False, fps=144):  # the with_display argument is True by default
        sim.visualizer.toggle_record()  # let's turn the recording on
        while sim.tick_count < 3000:  # let's step the physics engine 3 thousand times!
            sim.tick()




