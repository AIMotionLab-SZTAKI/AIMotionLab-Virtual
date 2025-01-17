import os
import sys
import pathlib
import numpy as np
import math
import yaml
import copy
import mujoco as mj

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
from mpcc_car import MPCCCar as Car
from trajectory_util import Trajectory_Marker
from aiml_virtual.trajectory.car_trajectory import CarTrajectory
from car_mpc_controller import F1TENTHMJPCController

def quaternion_from_z_rotation(rotation_z):
    w = math.cos(rotation_z / 2)
    x = 0
    y = 0
    z = math.sin(rotation_z / 2)

    return f"{w} {x} {y} {z}"

phi0 = 3.14 / 2
car_pos = np.array([1.5, 1.5, 0.05])
car_quat = quaternion_from_z_rotation(phi0)
path_points = np.array(
    [
        [0, 0],
        [1.5, 1.5],
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

def create_control_model(c_pos, c_quat):
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=os.path.join("xml_models", "control_scene.xml"))

    c = Car()
    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)
    sim = simulator.Simulator(scn)

    return sim.model, sim.data, scn.xml_name

def load_mpcc_params(filename = "control_params.yaml"):
    with open(filename, 'r') as file:
        params = yaml.full_load(file)
        return params

if __name__ == "__main__":
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename="scene.xml")
    traj = CarTrajectory()
    traj.build_from_points_const_speed(path_points, path_smoothing=0.01, path_degree=4, const_speed=1.5)
    c = Car(has_trailer=False)
    scn.add_object(c, pos=f"{car_pos[0]} {car_pos[1]} {car_pos[2]}", quat=car_quat)
    m = Trajectory_Marker(x=path_points[:, 0], y=path_points[:, 1])
    params = load_mpcc_params()
    c.trajectory = traj
    scn.add_object(m)
    sim = simulator.Simulator(scn)
    control_model = copy.deepcopy(sim.model)
    control_data = copy.deepcopy(sim.data)
    controller = F1TENTHMJPCController(control_model, control_data, trajectory=traj, params=params)
    c.controller = controller
    qpos0 = np.zeros(c.model.nq)
    qpos0[:3] = car_pos
    qpos0[3] = 3.14 / 2
    with sim.launch():
        mj.mju_copy(c.data.qpos, qpos0)
        while sim.viewer.is_running():
            sim.tick()
