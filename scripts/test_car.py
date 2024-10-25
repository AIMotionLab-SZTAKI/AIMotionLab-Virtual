import os
import sys
import pathlib
import numpy as np
"""
CHECK OUT original/scripts/test_car_controller.py
"""

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

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parents[1].resolve().as_posix()
    xml_directory = os.path.join(project_root, "xml_models")
    scene = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"))

    traj = car_trajectory.CarTrajectory()
    traj.build_from_points_smooth_const_speed(path_points=path_points, path_smoothing=0.01, path_degree=4, virtual_speed=1)

    # traj = car_trajectory.CarTrajectorySpatial()
    # traj.build_from_points_const_speed(path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5, start_delay=2)

    c = car.Car()
    c.trajectory = traj
    scene.add_object(c, pos="0 0 0.052", quat='0.9485664043524404 0 0 0.31657823130133655')
    sim = simulator.Simulator(scene)
    with sim.launch():
        while sim.viewer.is_running():
            sim.tick()