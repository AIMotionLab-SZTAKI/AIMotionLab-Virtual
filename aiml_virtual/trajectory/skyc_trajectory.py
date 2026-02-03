"""
This module contains the classes responsible for reading and evaluating a trajectory read from a skyc file.
"""

from typing import Any, Optional
import numpy as np
from skyc_utils.trajectory import Trajectory as sTrajectory
from skyc_utils import skyc, light_program
from aiml_virtual.trajectory import trajectory

class SkycTrajectory(trajectory.Trajectory):
    """
    Class that evaluates a trajectory read from a skyc file. Skyc trajectories
    consist of Bezier curves, represented in the appropriate trajectory.json file, selected by the aforementioned ID.
    In this trajectory.json there will be a "points" list, where each element describes a Bezier curve like so:

    - A float: this is the end time of the curve.
    - A list of  four floats (x-y-z-yaw): this is the physical end of the curve.
    - A variable length list of elements like the previous one: these are the (optional) extra control points.

    A Bezier curve is described by its parameter and its control points. The first and last control points are the
    beginning and end of said curve, so in a piecewise representation, if we stored each point for each individual
    segment, the end time and end pose of the Nth semgnet matches the start time and pose of the N-1th segment, leading
    to redundancy in the representation. Removing this redundancy is the rationale behind the described trajectory.json.
    This means that if "points" is of length N, there will be only N-1 segments in the trajectory. To this end, the 0th
    segment is special (it has to be, since it has no previous segment to describe its starting time/point): it shall
    have no extra control points.
    """
    def __init__(self, traj: sTrajectory, lights: Optional[light_program.LightProgram] = None) -> None:
        super().__init__()
        self.sTraj: sTrajectory = traj #: The skyc trajectory to evaluate.
        self.light_data: light_program.LightProgram = lights #: The light program to evaluate alongside the trajectory.
        self.start_time = 0 #: The time when the trajectory starts.

    def evaluate(self, time: float) -> dict[str, Any]:
        """
        Overrides (implements) superclass' evaluate. If the timestamp is before or after the trajectory, returns the
        start or end of the trajectory respectively, with the derivatives set to 0.

        Args:
            time (float): The timestamp when the trajectory must be evaluated.

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        time = time - self.start_time if self.started else 0
        retval = {
            "load_mass": 0.0,
            "target_pos": self.sTraj.evaluate(time).pose[:3],
            "target_vel": self.sTraj.evaluate(time).vel[:3],
            "target_acc": self.sTraj.evaluate(time).acc[:3],
            "target_rpy": np.array([0, 0, self.sTraj.evaluate(time).pose[3]]),
            "target_ang_vel": np.array([0, 0, self.sTraj.evaluate(time).vel[3]]),
        }
        return retval

def extract_trajectories(skyc_file: str) -> list[SkycTrajectory]:
    """
    Creates a list of SkycTrajectories from a skyc file.

    Args:
        skyc_file (str): The string of the skyc file's path.

    Returns:
        list[SkycTrajectory]: A list with one SkycTrajectory per drone in the skyc file.
    """
    parsed_skyc = skyc.Skyc.from_file(skyc_file)
    if parsed_skyc.has_lights:
        return [SkycTrajectory(traj, lights) for traj, lights in parsed_skyc.drones]
    else:
        return [SkycTrajectory(traj) for traj, in parsed_skyc.drones]