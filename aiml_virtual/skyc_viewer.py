"""
Module to visualize .skyc trajectory files.
"""

import os
from typing import Optional
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from functools import partial
from skyc_utils.skyc import plot_skyc_trajectories

import aiml_virtual
from aiml_virtual.scene import Scene
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.simulated_object.mocap_object.mocap_drone.mocap_crazyflie import MocapCrazyflie
from aiml_virtual.simulated_object.simulated_object import SimulatedObject
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory, extract_trajectories
from aiml_virtual.utils.utils_general import quaternion_from_euler
from aiml_virtual.simulator import Simulator, Event
from aiml_virtual.mocap.skyc_mocap_source import SkycMocapSource
from aiml_virtual.utils.utils_general import warning

def _pick_skyc_file() -> str:
    """
    Open a file picker dialog to select a .skyc file.
    If GUI is not available, prompt for input in the terminal.

    Returns:
        str: The path to the selected .skyc file.
    """
    # Compute starting directory: parent of the aiml_virtual package folder
    aiml_dir = os.path.dirname(os.path.abspath(aiml_virtual.__file__))
    start_dir = os.path.dirname(aiml_dir)

    # Try a GUI file dialog first
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            initialdir=start_dir,
            title="Select a .skyc file",
            filetypes=[("Skyc files", "*.skyc"), ("All files", "*.*")]
        )
        root.update()
        root.destroy()
        if not path:
            raise RuntimeError("No file selected.")
        return path
    except Exception:
        # Fallback: prompt in the terminal
        print(f"Please enter path to a .skyc file (starting at: {start_dir})")
        path = input("> ").strip()
        if not path:
            raise RuntimeError("No file provided.")
        return path

class ViewerCrazyflie(Crazyflie):
    """
    Crazyflie with a semi-transparent sphere to visualize its position, and color changes.
    """
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(
            drone, "geom", name=self.name+"_sphere", type="sphere", size="0.1",
            rgba=f"{r} {g} {b} {0.2}", contype="0", conaffinity="0"
        )
        return ret

class ViewerMocapCrazyflie(MocapCrazyflie):
    """
    Mocap Crazyflie with a semi-transparent sphere to visualize its position, and color changes.
    """
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(drone, "geom", name=self.name + "_sphere", type="sphere", size="0.1", rgba=f"{r} {g} {b} {0.2}",
                      contype="0", conaffinity="0")
        return ret

def close_pairs_by_xpos(objs: list[SimulatedObject], r: float) -> list[tuple[SimulatedObject, SimulatedObject]]:
    """
    Find all pairs of objects that are closer than r based on their xpos attribute.
    Returns [(obj_i, obj_j), ...] for all pairs with ||xpos_i - xpos_j|| < r.

    Args:
        objs (list[SimulatedObject]): List of simulated objects with xpos attribute.
        r (float): Distance threshold.

    Returns:
        list[tuple[SimulatedObject, SimulatedObject]]: List of pairs of objects closer than r.
    """
    n = len(objs)
    if n < 2:
        return []
    P = np.vstack([np.asarray(o.xpos, dtype=float) for o in objs])  # (n,3)
    pairs_idx = cKDTree(P).query_pairs(r)  # set of (i,j)
    return [(objs[i], objs[j]) for (i, j) in pairs_idx]


class SkycViewer:
    """
    Class to visualize .skyc trajectory files in the aiml_virtual simulator.
    """
    def __init__(self,
                 skyc_file: Optional[str] = None, # Path to the .skyc file. If None, a file picker dialog will open.
                 delay: float = 0.0, # Delay in seconds before starting the trajectories.
                 log_collisions: bool = True # Whether to log collisions between crazyflies.
                 ):
        self.log_collisions: bool = log_collisions #: Whether to log collisions between crazyflies.
        if not skyc_file:
            skyc_file = _pick_skyc_file()

        # Optionally normalize relative paths (useful if picking from skyc_viewer subpackage)
        self.skyc_file = os.path.abspath(skyc_file)

        self.trajectories: list[SkycTrajectory] = extract_trajectories(self.skyc_file) #: List of SkycTrajectories extracted from the .skyc file.
        for traj in self.trajectories:
            traj.start_time = delay
            traj.started = delay < 0.001
        self.crazyflies: list[ViewerCrazyflie | ViewerMocapCrazyflie] = [] #: List of crazyflies in the simulation.
        self.scn: Optional[Scene] = None #: Scene for the simulation.
        self.sim: Optional[Simulator] = None #: Simulator instance.

    def plot(self) -> None:
        """
        Plot the trajectories using skyc_utils' plotting function.
        """
        plot_skyc_trajectories(self.skyc_file)

    def _play(self, speed: float):
        """
        Internal method to run the simulation loop.

        Args:
            speed (float): Speed multiplier for the simulation.
        """
        with self.sim.launch(speed=speed):
            while not self.sim.display_should_close():
                self.sim.tick()
                if self.log_collisions:
                    collision_pairs = close_pairs_by_xpos(self.crazyflies, 0.2)
                    for a, b in collision_pairs:
                        warning(f"[{self.sim.sim_time:.2f}] COLLISION BETWEEN {a.name} and {b.name}")

    def play_raw(self, speed: float = 1.0) -> None:
        """
        Play the trajectories using raw mocap data, meaning that the crazyflies are mocap objects, following
        the trajectory exactly as it's defined.

        Args:
            speed (float): Speed multiplier for the simulation.
        """
        self.scn = Scene(os.path.join(aiml_virtual.xml_directory, "empty_checkerboard.xml"))
        self.sim = Simulator(self.scn)
        mocap = SkycMocapSource(self.trajectories, lambda: self.sim.sim_time)
        self.crazyflies = self.scn.add_mocap_objects(mocap)
        for mocap_name, traj in mocap.trajectories.items():
            cf = next(item for item in self.crazyflies if item.mocap_name == mocap_name)
            for t, color in traj.light_data.colors:
                r, g, b = color.as_list()
                self.sim.add_event(Event(
                    t + traj.start_time,
                    partial(cf.set_color, r / 255, g / 255, b / 255, 0.2)  # <- captures cf,r,g,b
                ))
        self._play(speed)

    def play_with_controller(self, speed: float = 1.0) -> None:
        """
        Play the trajectories using crazyflies with controllers, meaning that the crazyflies are controlled
        objects using a geom controller to follow the trajectory.

        Args:
            speed (float): Speed multiplier for the simulation.
        """
        self.scn = Scene(os.path.join(aiml_virtual.xml_directory, "empty_checkerboard.xml"))
        self.sim = Simulator(self.scn)
        for traj in self.trajectories:
            cf = ViewerCrazyflie()
            cf.trajectory = traj
            start = traj.sTraj.start
            for t, color in traj.light_data.colors:
                r, g, b = color.as_list()
                self.sim.add_event(Event(
                    t + traj.start_time,
                    partial(cf.set_color, r/255, g/255, b/255, 0.2)  # <- captures cf,r,g,b
                ))

            quat = quaternion_from_euler(0, 0, start.yaw)
            self.scn.add_object(
                cf,
                f"{start.x} {start.y} {start.z}",
                f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "0.5 0.5 0.5 1"
            )
            self.crazyflies.append(cf)
        self._play(speed)