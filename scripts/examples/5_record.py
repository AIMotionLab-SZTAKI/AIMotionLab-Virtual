"""
This script shows how recording works and how you can "save" the frame of a mocap source.
"""

import os
import sys
import pathlib
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
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename="example_scene_5.xml")
    # A common use case is the need to "save" the position of mocap objects in a mocap stream at a given time.
    # This can be done using the DummyMocapSource.freeze(other_mocap) function, which returns a new MocapSource
    # that always contains the frame that was present in other_mocap at the time of initialization.
    dummy_mocap_dict = {
        "airport": (np.array([-1, 1, 0]), np.array([0, 0, 0, 1])),
        "parking_lot_0": (np.array([0, 1, 0]), np.array([0, 0, 0, 1])),
        "landing_zone_10": (np.array([1, 1, 0]), np.array([0, 0, 0, 1])),
        "obst1": (np.array([-1, 0, 0]), np.array([0, 0, 0, 1])),
        "payload 12": (np.array([1, 0, 0]), np.array([0, 0, 0, 1])),
        "bu1": (np.array([-1, -1, 0]), np.array([0, 0, 0, 1])),
        "bu2": (np.array([0, -1, 0]), np.array([0, 0, 0, 1])),
        "bu3": (np.array([1, -1, 0]), np.array([0, 0, 0, 1])),
        "MocapBumblebee": (np.array([0, 0, 0.5]), np.array([0, 0, 0, 1]))
    }
    frame_generator = partial(dummy_mocap_source.generate_circular_paths, start_poses=dummy_mocap_dict, T=5)
    # mocap1 moves the objects above around in a circle
    mocap1 = dummy_mocap_source.DummyMocapSource(frame_generator=frame_generator)
    # mocap2 is a snapshot of mocap1
    mocap2 = dummy_mocap_source.DummyMocapSource.freeze(mocap1)
    scn.add_mocap_objects(mocap2)
    # Let's add a moving mocap drone connected to mocap1 as well:
    bb = mocap_bumblebee.MocapBumblebee(mocap1, "MocapBumblebee")
    scn.add_object(bb, color="0 1 0 1")
    sim = simulator.Simulator(scn)  # note that render fps is separate from display fps!
    with sim.launch(fps=30, renderer_fps=120):  # note that render fps is separate from display fps!
        # Let's say you'd like to record a video of your simulation. The simulator has an optional rendering process that
        # you may toggle on or off. Whenever it's toggled ON, the simulator saves the frames it renders and upon closing
        # the simulator, a video is created. This process is toggled OFF when the simulator is initialized, but you can
        # toggle it ON or OFF using the shift+R keybind from the simulation window, or set it via code like so:
        sim.processes["render"].resume()  # this way the simulator records by default
        # Note that where the video gets saved under cwd/simulator.mp4 by default.
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses





