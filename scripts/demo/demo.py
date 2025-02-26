import os
import sys
import pathlib
from xml.etree import ElementTree as ET

import numpy as np
from skyc_utils.skyc_maker import write_skyc, XYZYaw, Trajectory, TrajectoryType, LightProgram, Color
from skyc_utils.skyc_inspector import get_traj_data
import socket
import threading
from typing import Optional, Union
import copy
from scipy import interpolate
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

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
from aiml_virtual.mocap.mocap_source import MocapSource
from aiml_virtual.simulated_object.mocap_object.mocap_object import Box
from Utils.Utils import load
from skyc_utils.skyc_maker import trim_ppoly, extend_ppoly_coeffs
from aiml_virtual.simulated_object.mocap_object.mocap_drone.mocap_drone import MocapDrone
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.simulated_object.mocap_object.mocap_drone.mocap_crazyflie import MocapCrazyflie

class DemoCrazyflie(Crazyflie):
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "DemoCrazyflie"

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(drone, "geom", name=self.name+"_sphere", type="sphere", size="0.1", rgba=f"{r} {g} {b} {0.2}",
                      contype="0", conaffinity="0")
        return ret

class DeMocapCrazyflie(MocapCrazyflie):
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "DeMocapCrazyflie"

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(drone, "geom", name=self.name+"_sphere", type="sphere", size="0.1", rgba=f"{r} {g} {b} {0.2}",
                      contype="0", conaffinity="0")
        return ret

for k, v in MocapSource.config.items():
    if v=="MocapCrazyflie":
        MocapSource.config[k] = "DeMocapCrazyflie"

#TODO: Comments, docstrings
COLORS = [Color.GREEN, Color.BLUE, Color.CYAN, Color.MAGENTA, Color.YELLOW]

def generate_skyc(input_file: str, output_file: str, add_colors: bool = False):
    """
    Generates a skyc file named output_file.skyc from the routes read from input_file
    """
    routes: list[list[dict[str, Union[int, np.ndarray]]]] = load(input_file)
    # routes will have one element per drone
    # that element will be again a list, which contains one element per trajectory
    # that element is a dict, where the key 'Route' shows the points the trajectory
    # those points are stored in a numpy array, where the columns are t-x-y-z-vel
    # in the resulting trajectory, we'd like to have no more than 60 segments altogether, which may prove a
    # challenge as there will be more than 60 points per drone.
    trajectories = []
    for drone_segments in routes:
        starts = [segment['Start'] for segment in drone_segments]
        goals = starts[1:]+starts[:1]
        for i, goal in enumerate(goals):
            drone_segments[i]['Start'] = goal
        # drone_segments is a list of dicts containing a 'Route' element which is a np array of the trajectory's points
        takeoff_time = 2
        # the 0th segments actually start at 1sec, so the takeoff will take 3 seconds instead of 2
        for segment in drone_segments: # push all trajectories back by takeoff_time
            segment['Route'][:, 0] += takeoff_time
        first_point = copy.deepcopy(np.append(drone_segments[0]['Route'][0][1:4], 0))
        first_point[2] = 0.0
        start = XYZYaw(first_point) # the start = takeoff position is under the first point of the first trajectory
        traj = Trajectory(TrajectoryType.POLY4D, degree=5)
        traj.set_start(start)
        # 62 segments can be stored in the trajectory memory. One will be the landing segment, one the takeoff segment,
        # meaning that we may use 60 for all segments. One "position hold" is necessary between segments, hende:
        bezier_curve_per_segment = int(60.0 / len(drone_segments)) - 1
        last_time = 0
        for segment in drone_segments:
            # a segment is a dict, where 'Start' signals the node where it starts from, and 'Route' the points it
            # touches along the way.
            points = segment["Route"]
            dt = points[0][0] - last_time
            # start off with a position hold if there is a delay between the last route's end and the next one
            traj.add_goto(XYZYaw(np.append(points[0][1:4], 0)), dt)
            last_time = points[-1][0]
            points[:, -1] = 0
            t_x_y_z_yaw = points.T.tolist()
            # When adding an interpolated segment, the segment is supposed to start at time 0.
            t_x_y_z_yaw[0] = [timestamp - t_x_y_z_yaw[0][0] for timestamp in t_x_y_z_yaw[0]]
            if len(points) > bezier_curve_per_segment:
                traj.add_interpolated_traj(t_x_y_z_yaw, number_of_segments=bezier_curve_per_segment, method="mosek",
                                           fit_ends=True, force_0_derivs=True)
            else:
                # if there are fewer or equal points in the trajectory than the resulting number of bezier segments,
                # we don't need to bother with optimizing the nodes using mosek, we can just use a spline fitting,
                # which will result in the exact same number of segments as the number of points
                t = t_x_y_z_yaw[0]
                bc = ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)])  # boundary conditions: start and end with 0 derivs
                bspline_lst = [interpolate.make_interp_spline(t, values, k=5, bc_type=bc) for values in
                               t_x_y_z_yaw[1:]]
                # remove duplicate nodes at the ends
                trimmed_ppoly_lst = [trim_ppoly(interpolate.PPoly.from_spline(bspline)) for bspline in bspline_lst]
                for ppoly in trimmed_ppoly_lst:
                    extend_ppoly_coeffs(ppoly, 6)
                traj.add_ppoly(XYZYaw(trimmed_ppoly_lst))
        traj.add_goto(start, takeoff_time)
        traj.add_parameter(-15, "stabilizer.controller", 2)
        trajectories.append(traj)
    if add_colors:
        light_programs = []
        for drone_segments in routes:
            light_program = LightProgram()
            last_time = 3
            light_program.append_color(Color.WHITE, last_time)
            for segment in drone_segments:
                color = COLORS[segment["Start"]%len(COLORS)]
                points = segment["Route"]
                duration = points[-1][0] - last_time
                duration = round(duration*50)/50
                last_time = points[-1][0]
                light_program.append_color(color, duration)
            light_programs.append(light_program)
        write_skyc(trajectories, light_programs, name=output_file)
    else:
        write_skyc(trajectories, name=output_file)

def start_show(soc: socket.socket, virt_trajectories: list[SkycTrajectory], real_trajectories: list[SkycTrajectory],
               simulator: Simulator, mocap: OptitrackMocapSource) -> None:
    """
    Waits for a b"START" bytearray from the server, and doesn't return until it receives it. When that happens,
    the trajectories of the drones are started, and the drones are turned green to indicate it.
    """
    while True:
        res = soc.recv(1024).strip()
        if res == b"START":
            print("START SIGNAL RECEIVED")
            break
    def d(p1: np.ndarray, p2: np.ndarray):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx*dx + dy*dy)
    traj_dict: dict[str, SkycTrajectory] = {}
    # start_positions = [t.evaluate(0)["target_pos"][:2] for t in real_trajectories]
    data = mocap.data
    while len(data) == 0:
        print("No objects found in mocap, sleeping zzzzZZZZZzz")
        time.sleep(0.5)
    for traj in virt_trajectories:
        traj.set_start(simulator.sim_time)
    for obj in simulator.simulated_objects:
        if isinstance(obj, MocapDrone):
            name = obj.mocap_name
            pos = data[name][0]
            for traj in real_trajectories:
                start_position = traj.evaluate(0)["target_pos"][:2]
                # TODO: CHECK IF CORRECT
                if name not in traj_dict or d(pos, start_position) < d(pos, traj_dict[name].evaluate(0)["target_pos"][:2]):
                    traj_dict[name] = traj
            print(f'Distance from start point: {d(pos, traj_dict[name].evaluate(0)["target_pos"][:2])}')
            t = threading.Thread(target=handle_colors, args=(obj, traj_dict[name].light_data), daemon=True)
            t.start()

def handle_colors(drone: MocapDrone, light_data: dict):
    for color, duration in light_data['colors']:
        r, g, b = map(int, color.split(','))
        drone.set_color(r/255, g/255, b/255, 0.2)
        time.sleep(duration)

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 6002  # 6002 is actual server port, 7002 is Dummy Port
    generate_skyc("Saves/Routes/virtual_routes", "virtual")
    generate_skyc("Saves/Routes/real_routes", "real", add_colors=True)
    scene = Scene(os.path.join(xml_directory, "demo_base.xml"))
    virtual_trajectories = extract_trajectories("virtual.skyc")
    real_trajectories = extract_trajectories("real.skyc")
    for trajectory in virtual_trajectories:
        cf = DemoCrazyflie()
        cf.trajectory = trajectory
        start_pos = trajectory.evaluate(0.0).get("target_pos")
        scene.add_object(cf, f"{start_pos[0]} {start_pos[1]} {start_pos[2]}", "1 0 0 0", "1 0 0 1")
    mocap = OptitrackMocapSource()
    scene.add_mocap_objects(mocap)
    # TODO: change this to load from a file
    building_data  = load("Saves/Scenarios/city_scenario")["Scenario"].static_obstacles.enclosed_spaces[-8:]
    for x, y, h, w, l in building_data:
        scene.add_object(Box(w/2, w/2, h), f"{x} {y} {0}")
    simulator = Simulator(scene)
    try:
        soc = socket.socket()
        soc.connect((ip, port))
        thread = threading.Thread(target=start_show, args=(soc, virtual_trajectories, real_trajectories, simulator, mocap), daemon=True)
        thread.start()
        with simulator.launch():
            while not simulator.display_should_close():
                simulator.tick()
    except ConnectionRefusedError:
        print("Server connection refused.")

