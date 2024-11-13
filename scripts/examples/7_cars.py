"""
This script shows how you can add cars to the scene.
"""

import os
import sys
import pathlib
import numpy as np

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
from aiml_virtual.simulated_object.dynamic_object.controlled_object import car
from aiml_virtual.trajectory import car_trajectory

path_points = np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 2],
        [4, 1],
        [4.5, 0],
        [4, -1],
        [3, -2],
        [2, -2],
        [1, -1],
        [0, 0],
        [-1, 1],
        [-2, 2],
        [-3, 2],
        [-4, 1],
        [-4.5, 0],
        [-4, -2.1],
        [-3, -2.3],
        [-2, -2],
        [-1, -1],
        [0, 0],
    ]
)
path_points1 = path_points + np.array([1.5, 0])
path_points2 = path_points + np.array([-1.5, 0])

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename="example_scene_7.xml")
    # So far dynamic objects only had default constructors, but when it comes to Cars, you can provide a boolean
    # argument to the constructor to decide whether to add a trailer.
    car1 = car.Car()
    traj1 = car_trajectory.CarTrajectory()
    # This is the simplest way to build a Car trajectory: provide the points for it, and build with const speed
    traj1.build_from_points_smooth_const_speed(path_points=path_points1, path_smoothing=0.1, path_degree=4, virtual_speed=1)
    car1.trajectory = traj1

    # For the next car, let's add a trailer
    car2 = car.Car(has_trailer=True)
    # A Spatial car trajectory ise parametrized by arc length
    traj2 = car_trajectory.CarTrajectorySpatial()
    traj2.build_from_points_const_speed(path_points=path_points2, path_smoothing=0.1, path_degree=4, const_speed=1)
    car2.trajectory = traj2

    scn.add_object(car1, pos="1.5 0 0.05", quat='0.948 0 0 0.3165')
    scn.add_object(car2, pos="-1.5 0 0.05", quat='0.948 0 0 0.3165')
    sim = simulator.Simulator(scn)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()  # tick steps the simulator, including all its subprocesses




