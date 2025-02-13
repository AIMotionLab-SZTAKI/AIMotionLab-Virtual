import os
import sys
import pathlib
import numpy as np
from skyc_utils.skyc_maker import write_skyc, XYZYaw, Trajectory, TrajectoryType
from skyc_utils.skyc_inspector import get_traj_data
import socket
import threading
from typing import Optional, Union
import copy
from scipy import interpolate

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
from aiml_virtual.scene import Scene
from aiml_virtual.simulator import Simulator
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory, extract_trajectories
from aiml_virtual.mocap.optitrack_mocap_source import OptitrackMocapSource
from Utils.Utils import load
from skyc_utils.skyc_maker import trim_ppoly, extend_ppoly_coeffs

def generate_skyc(input_file: str, output_file: str):
    routes: list[list[dict[str, Union[int, np.ndarray]]]] = load(input_file)
    # routes will have one element per drone
    # that element will be again a list, which contains one element per trajectory
    # that element is a dict, where the key 'Route' shows the points the trajectory
    # those points are stored in a numpy array, where the columns are t-x-y-z-vel
    # in the resulting trajectory, we'd like to have no more than 60 segments altogether, which will be a
    # challenge as there will be more than 60 points per drone
    trajectories = []
    for drone_segments in routes:
        takeoff_time = 2
        for segment in drone_segments:
            segment['Route'][:, 0] += takeoff_time
        # We want each drone to end up having 60 segments plus one takeoff segment
        first_point = copy.deepcopy(np.append(drone_segments[0]['Route'][0][1:4], 0))
        first_point[2] = 0.0
        start = XYZYaw(first_point)
        traj = Trajectory(TrajectoryType.POLY4D, degree=5)
        traj.set_start(start)
        bezier_curve_per_segment = int(60.0 / len(drone_segments)) - 1
        last_time = 0
        for segment in drone_segments:
            points = segment["Route"]
            dt = points[0][0] - last_time
            traj.add_goto(XYZYaw(np.append(points[0][1:4], 0)), dt)
            last_time = points[-1][0]
            points[:, -1] = 0
            t_x_y_z_yaw = points.T.tolist()
            t_x_y_z_yaw[0] = [t - t_x_y_z_yaw[0][0] for t in t_x_y_z_yaw[0]]
            if len(points) > bezier_curve_per_segment:
                traj.add_interpolated_traj(t_x_y_z_yaw, number_of_segments=bezier_curve_per_segment, method="mosek",
                                           fit_ends=True, force_0_derivs=True)
            else:
                t = t_x_y_z_yaw[0]
                bc = ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)])
                bspline_lst = [interpolate.make_interp_spline(t, values, k=5, bc_type=bc) for values in
                               t_x_y_z_yaw[1:]]
                trimmed_ppoly_lst = [trim_ppoly(interpolate.PPoly.from_spline(bspline)) for bspline in bspline_lst]
                for ppoly in trimmed_ppoly_lst:
                    extend_ppoly_coeffs(ppoly, 6)
                traj.add_ppoly(XYZYaw(trimmed_ppoly_lst))
        traj.add_goto(start, takeoff_time)
        trajectories.append(traj)
    write_skyc(trajectories, name=output_file)


def initialize_from_skyc(skyc_file: str, base: str, save_filename: str) -> Scene:
    scn = Scene(os.path.join(xml_directory, base), save_filename=save_filename)
    traj_data = get_traj_data(skyc_file)
    for i in range(len(traj_data)):
        cf = Crazyflie()
        traj = SkycTrajectory(skyc_file, i)
        cf.trajectory = traj
        start_pos = traj.evaluate(0.0).get("target_pos")
        scn.add_object(cf, f"{start_pos[0]} {start_pos[1]} {start_pos[2]}", "1 0 0 0", "1 0 0 1")
    return scn

def wait_show_start(soc: socket.socket, trajectories: list[SkycTrajectory], simulator: Simulator):
    while True:
        res = soc.recv(1024).strip()
        if res == b"START":
            print("START SIGNAL RECEIVED")
            for trajectory in trajectories:
                trajectory.set_start(simulator.sim_time)
            break

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 7002  # 6002
    generate_skyc("Saves/Routes/virtual_routes", "virtual")
    generate_skyc("Saves/Routes/real_routes", "real")
    # scene = Scene(os.path.join(xml_directory, "scene_base.xml"))
    scene = Scene(os.path.join(xml_directory, "empty_checkerboard.xml"))
    trajectories = extract_trajectories("virtual.skyc")
    for trajectory in trajectories:
        cf = Crazyflie()
        cf.trajectory = trajectory
        start_pos = trajectory.evaluate(0.0).get("target_pos")
        scene.add_object(cf, f"{start_pos[0]} {start_pos[1]} {start_pos[2]}", "1 0 0 0", "1 0 0 1")
    mocap = OptitrackMocapSource()
    scene.add_mocap_objects(mocap)
    simulator = Simulator(scene)

    try:
        soc = socket.socket()
        soc.connect((ip, port))
        t = threading.Thread(target=wait_show_start, args=(soc, trajectories, simulator), daemon=True)
        t.start()
        with simulator.launch():
            while not simulator.display_should_close():
                simulator.tick()
    except ConnectionRefusedError:
        print("Server connection refused.")

