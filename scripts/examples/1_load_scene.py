"""
This script shows how to load and display a simulation from an xml.
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
    # A Scene is the container for the objects in a simulation. A Simulator needs a scene to simulate. To this end,
    # let's read a Scene from a mjcf file. This scene contains dynamic object, controlled objects and mocap objects
    # as well: all things that we will talk about in further examples.
    scn = scene.Scene(os.path.join(xml_directory, "example_scene_1.xml"), save_filename="example_scene_1.xml")
    # Once we have our scene, we can simulate it using a Simulator
    sim = simulator.Simulator(scn)
    # We can start displaying our simulation by launching its context handler.
    with sim.launch(fps=100):
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses
