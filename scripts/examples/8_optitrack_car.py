"""
As opposed to dynamic cars, currently the trailer and the car are separate mocap objects. This
example is meant to demonstrate that, as well as the usage of the Optitrack system as a mocap source.
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
project_root = project_root.resolve().as_posix()

from aiml_virtual import scene, simulator
from aiml_virtual.simulated_object.mocap_object.mocap_car import MocapCar, MocapTrailer
from aiml_virtual.mocap import optitrack_mocap_source

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "scene_base.xml"), save_filename="example_scene_8.xml")
    mocap = optitrack_mocap_source.OptitrackMocapSource()
    car = MocapCar(mocap, "JoeBush1")
    scn.add_object(car)
    trailer = MocapTrailer(mocap, "trailer")
    scn.add_object(trailer)
    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses