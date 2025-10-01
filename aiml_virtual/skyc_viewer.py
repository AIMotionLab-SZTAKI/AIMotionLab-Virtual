# TODO: COMMENTS, DOCSTRINGS
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
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(drone, "geom", name=self.name + "_sphere", type="sphere", size="0.1", rgba=f"{r} {g} {b} {0.2}",
                      contype="0", conaffinity="0")
        return ret

def close_pairs_by_xpos(objs: list[SimulatedObject], r: float) -> list[tuple[SimulatedObject, SimulatedObject]]:
    """Return [(obj_i, obj_j), ...] for all pairs with ||xpos_i - xpos_j|| < r."""
    n = len(objs)
    if n < 2:
        return []
    P = np.vstack([np.asarray(o.xpos, dtype=float) for o in objs])  # (n,3)
    pairs_idx = cKDTree(P).query_pairs(r)  # set of (i,j)
    return [(objs[i], objs[j]) for (i, j) in pairs_idx]

# TODO: comments and docstrings
class SkycViewer:
    def __init__(self,
                 skyc_file: Optional[str] = None,
                 graphs: bool = True,
                 delay: float = 0.0,
                 started: bool = True,
                 log_collisions: bool = True):
        self.log_collisions = log_collisions
        # NEW: make the skyc_file optional and open a picker if not provided
        if not skyc_file:
            skyc_file = _pick_skyc_file()

        # Optionally normalize relative paths (useful if picking from skyc_viewer subpackage)
        skyc_file = os.path.abspath(skyc_file)

        if graphs:
            plot_skyc_trajectories(skyc_file)

        self.trajectories: list[SkycTrajectory] = extract_trajectories(skyc_file)
        for traj in self.trajectories:
            traj.start_time = delay
            traj.started = started
        self.crazyflies: list[ViewerCrazyflie | ViewerMocapCrazyflie] = []
        self.scn = Scene(os.path.join(aiml_virtual.xml_directory, "empty_checkerboard.xml"))
        self.sim = Simulator(self.scn)

    def _play(self, speed: float):
        with self.sim.launch(speed=speed):
            while not self.sim.display_should_close():
                self.sim.tick()
                if self.log_collisions:
                    collision_pairs = close_pairs_by_xpos(self.crazyflies, 0.2)
                    for a, b in collision_pairs:
                        warning(f"[{self.sim.sim_time:.2f}] COLLISION BETWEEN {a.name} and {b.name}")

    def play_raw(self, speed: float = 1.0) -> None: # TODO: add colors here as well
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