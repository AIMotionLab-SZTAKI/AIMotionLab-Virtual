"""
This script shows how recording works and how you can "save" the frame of a mocap source.
"""

import os
import sys
import pathlib
import numpy as np
from functools import partial

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
from aiml_virtual.mocap import dummy_mocap_source
from aiml_virtual.simulated_object.mocap_object.mocap_drone import mocap_bumblebee

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename="example_scene_5.xml")
    # A common use case is the need to "save" the position of mocap objects in a mocap stream at a given time.
    # This can be done using the DummyMocapSource.freeze(other_mocap) function, which returns a new MocapSource
    # that always contains the frame that was present in other_mocap at the time of initialization.
    dummy_mocap_dict = {
        "Airport": (np.array([-1, 1, 0]), np.array([0, 0, 0, 1])),
        "ParkingLot": (np.array([0, 1, 0]), np.array([0, 0, 0, 1])),
        "LandingZone": (np.array([1, 1, 0]), np.array([0, 0, 0, 1])),
        "obst1": (np.array([-1, 0, 0]), np.array([0, 0, 0, 1])),
        "payload1": (np.array([1, 0, 0]), np.array([0, 0, 0, 1])),
        "bu11": (np.array([-1, -1, 0.9]), np.array([0, 0, 0, 1])),
        "bu13": (np.array([0, -1, 0]), np.array([0, 0, 0, 1])),
        "bu14": (np.array([1, -1, 0.2]), np.array([0, 0, 0, 1])),
        "bb0": (np.array([0, 0, 0.5]), np.array([0, 0, 0, 1]))
    }
    frame_generator = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_dict, T=5)
    # mocap1 moves the objects above around in a circle
    mocap1 = dummy_mocap_source.DummyMocapSource(frame_generator=frame_generator)
    # mocap2 is a snapshot of mocap1
    mocap2 = dummy_mocap_source.DummyMocapSource.freeze(mocap1)
    scn.add_mocap_objects(mocap2)
    # Let's add a moving mocap drone connected to mocap1 as well:
    bb = mocap_bumblebee.MocapBumblebee(mocap1, "bb0")
    scn.add_object(bb, color="0 1 0 1")
    sim = simulator.Simulator(scn)  # note that render fps is separate from display fps!
    with sim.launch(fps=60):
        # Let's say you'd like to record a video of your simulation. The simulator's visualizer has an attribute called
        # recording. Whenever a frame is generated, if this attribute is True, the frame gets saved to a video. This is
        # true even when there is no display.
        sim.visualizer.recording = True
        # Note that where the video gets saved under cwd/simulator.mp4 by default.
        while not sim.display_should_close():
            sim.tick()  # tick steps the simulator, including all its subprocesses





