"""
This script shows how to load and display a simulation from an xml.
"""

import os
import sys
import pathlib

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
# until we meet the "aiml_virtual" package, keep adding the current folder to the path and then changing folder into
# the parent folder
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())
xml_directory = os.path.join(project_root.resolve().as_posix(), "xml_models")

from aiml_virtual import scene, simulator

if __name__ == "__main__":
    # A Scene is the container for the objects in a simulation. A Simulator needs a scene to simulate. To this end,
    # let's read a Scene from a mjcf file.
    scn = scene.Scene(os.path.join(xml_directory, "Scene.xml"))
    # Once we have our scene, we can simulate it using a Simulator, which requires a target update frequency for
    # controllers to adhere to (default is 500), and a target fps for display to adhere to (default is 100).
    sim = simulator.Simulator(scn, update_freq=500, target_fps=100)
    # We can start displaying our simulation by launching the viewer as a context handler.
    with sim.launch_viewer():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses
