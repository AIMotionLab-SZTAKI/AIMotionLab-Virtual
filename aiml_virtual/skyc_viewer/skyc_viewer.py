import os
from typing import Optional
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
import mujoco

from skyc_utils.skyc import plot_skyc_trajectories

import aiml_virtual
from aiml_virtual.scene import Scene
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone.crazyflie import Crazyflie
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory, extract_trajectories
from aiml_virtual.utils.utils_general import quaternion_from_euler
from aiml_virtual.simulator import Simulator

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
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "ViewerCrazyflie"

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        ret = super().create_xml_element(pos, quat, color)
        drone = ret["worldbody"][0]
        r, g, b, _ = color.split()
        ET.SubElement(
            drone, "geom", name=self.name+"_sphere", type="sphere", size="0.1",
            rgba=f"{r} {g} {b} {0.2}", contype="0", conaffinity="0"
        )
        return ret

def close_pairs_by_xpos(objs: list[ViewerCrazyflie], r: float) -> list[tuple[ViewerCrazyflie, ViewerCrazyflie]]:
    """Return [(obj_i, obj_j), ...] for all pairs with ||xpos_i - xpos_j|| < r."""
    n = len(objs)
    if n < 2:
        return []
    P = np.vstack([np.asarray(o.xpos, dtype=float) for o in objs])  # (n,3)
    pairs_idx = cKDTree(P).query_pairs(r)  # set of (i,j)
    return [(objs[i], objs[j]) for (i, j) in pairs_idx]

# TODO: comments and docstrings
class SkycViewer:
    def __init__(self, skyc_file: Optional[str] = None, clearance: float = 0.2):
        # NEW: make the skyc_file optional and open a picker if not provided
        if not skyc_file:
            skyc_file = _pick_skyc_file()

        # Optionally normalize relative paths (useful if picking from skyc_viewer subpackage)
        skyc_file = os.path.abspath(skyc_file)

        self.trajectories: list[SkycTrajectory] = extract_trajectories(skyc_file)
        plot_skyc_trajectories(skyc_file)

    def play(self, delay: float = 1.0):
        scn = Scene(os.path.join(aiml_virtual.xml_directory, "empty_checkerboard.xml"))
        crazyflies = []
        for traj in self.trajectories:
            cf = ViewerCrazyflie()
            traj.time_offset = delay
            cf.trajectory = traj
            start = traj.traj.start
            quat = quaternion_from_euler(0, 0, start.yaw)
            scn.add_object(
                cf,
                f"{start.x} {start.y} {start.z}",
                f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "0.5 0.5 0.5 1"
            )
            crazyflies.append(cf)
        sim = Simulator(scn)

        with sim.launch():
            while not sim.display_should_close():
                sim.tick()
                for cf in crazyflies:
                    cf.set_color(0.5, 0.5, 0.5, 0.2)
                collision_pairs = close_pairs_by_xpos(crazyflies, 0.2)
                for a, b in collision_pairs:
                    a.set_color(0.5, 0, 0, 0.2)
                    b.set_color(0.5, 0, 0, 0.2)
                    print(f"WARNING: COLLISION BETWEEN {a.name} and {b.name}")
