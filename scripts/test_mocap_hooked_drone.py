import os
import sys
import pathlib
import numpy as np

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
from aiml_virtual.mocap import optitrack_mocap_source
from aiml_virtual.simulated_object.mocap_skeleton import mocap_hooked_bumblebee


if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "scene_base.xml"))
    mocap = optitrack_mocap_source.OptitrackMocapSource()
    bb = mocap_hooked_bumblebee.MocapHookedBumblebee2DOF(mocap, "bb3", "hook12")
    scn.add_object(bb)
    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()