"""
This script shows how mocap objects work.
"""

import os
import sys
import pathlib
import time

import numpy as np
from functools import partial

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
from aiml_virtual.mocap import dummy_mocap_source
from aiml_virtual.simulated_object.mocap_object.mocap_drone import mocap_bumblebee

if __name__ == "__main__":
    # As mentioned in 1_load_scene.py, we can load previously created scenes from xmls in order to continue working on
    # them. Let's do that here and use the scene created in 2_build_scene.py:
    scn = scene.Scene(os.path.join(xml_directory, "example_scene_2.xml"), save_filename=f"example_scene_4.xml")

    # As mentioned in 2_build_scene.py, mocap objects receive their pose information from a motion capture source,
    # rather than MuJoCo calculations. To this end let's create a motion capture source! The most versatile MocapSource
    # is a DummyMocapSource, which takes a frame generator as an argument. This frame generator should be a callable
    # with a header that takes no arguments and returns a dictionary containing the mocap data for a frame. An example
    # of a frame may look like this (a frame generator should return something looking like this):
    dummy_mocap_start_poses = {  # name: (position, quaterion)
        "cf0": (np.array([1, -1, 0.5]), np.array([0, 0, 0, 1])),
        "cf1": (np.array([1, 1, 0.5]), np.array([0, 0, 1, 0])),
        "bb0": (np.array([-1, 1, 0.5]), np.array([0, 1, 0, 0])),
        "bb1": (np.array([-1, -1, 0.5]), np.array([1, 0, 0, 0]))
    }
    # In order to make a frame generator, we can use a function provided in the dummy_mocap_source module.
    # This way, frame generator will generate mocap data that moves the elements of this dictionary around in a circle.
    frame_generator = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_start_poses, T=5)
    mocap = dummy_mocap_source.DummyMocapSource(frame_generator)
    # Now we have a fully primed MocapSource. Another alternative to get there would be an OptitrackMocapSource.
    # MocapSources run in their own thread in order to not impede simulation. They refresh their internal frame at
    # regular intervals, and objects subscribed to a mocap source read the frame whenever they want to.
    # A MocapObject can only be subscribed to one mocap source at a time. It will check the mocap source's frame for
    # its mocap_name, and update its pose in MuJoCo accordingly.
    # There are two ways to make MocapObjects:
    # (1): Adding all objects found in a MocapSource
    # (2): Initializing a mocap object with its MocapSource and mocap_name, then adding it with scene.add_object
    #      Note, that if there already are mocap objects in the scene that are connected to the given MocapSource,
    #      then this method will first remove them to avoid duplication.
    # Let's use method (1) first: this also returns a list of references to the added objects
    lst = scn.add_mocap_objects(mocap, "1 0 0 1")
    # At this point let's check another way to manipulate the scene: removing objects. This can be done either by
    # reference or by name. Let's remove an object by name:
    scn.remove_object("MocapBumblebee_0")
    # And then let's remove an object by reference:
    scn.remove_object(scn.simulated_objects[-1])
    # In their stead, let's add a MocapBumblebee using method (2):
    bb0 = mocap_bumblebee.MocapBumblebee(mocap, "bb0")  # In the MocapSource, this bb will look for "bb0"
    scn.add_object(bb0)
    sim = simulator.Simulator(scn, update_freq=500, target_fps=100)
    with sim.launch_viewer():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses