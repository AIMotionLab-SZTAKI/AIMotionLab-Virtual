import os
import sys
import pathlib

import glfw
import numpy as np

project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix())  # add the folder this file is in to path
while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual
xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator
from aiml_virtual.trajectory import dummy_drone_trajectory
from aiml_virtual.simulated_object.dynamic_object import dynamic_object
from aiml_virtual.simulated_object.dynamic_object.controlled_object import bicycle

if __name__ == "__main__":
    # good one bro
    # The bug that bugs me: if i run two independent simulations, both with display (or both without display),
    # everything works fine. But if I run one
    # simulation with display, and then another without display, the second one crashes.
    # So far, the only workaround I've found is to call glfw.terminate() in between the two simulations. But still then,
    # a GLFW error is printed to the console.


    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_3.xml")
    scn.add_object(bicycle.Bicycle(), "0 1 0.1", "1 0 0 0", "0.5 0.5 0.5 1")
    sim = simulator.Simulator(scn)
    with sim.launch(with_display=True):
        while not sim.display_should_close() and sim.data.time < 2:
            sim.tick()

    scn_2 = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"example_scene_3.xml")
    scn_2.add_object(bicycle.Bicycle(), "0 1 0.1", "1 0 0 0", "0.5 0.5 0.5 1")
    sim_2 = simulator.Simulator(scn_2)
    with sim_2.launch(with_display=False):
        while not sim_2.display_should_close() and sim_2.data.time < 2:
            sim_2.tick()




