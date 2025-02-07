"""
This module contains the classes responsible for reading and evaluating a trajectory read from a skyc file.
"""

from typing import Any, Optional
import numpy as np
from skyc_utils import skyc_inspector

from aiml_virtual.trajectory import trajectory

BezierCurve = skyc_inspector.BezierCurve
TrajEvaluator = skyc_inspector.TrajEvaluator

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
    def __init__(self, traj_data: dict[str, Any]):
        super().__init__()
        self.traj_data: dict[str, Any] = traj_data  #: The dictionary that can be read from trajectory.json.
        self.traj_data["started"] = False
        self.evaluator: Optional[TrajEvaluator] = None

    def set_start(self, t: float):
        """
        Sets the start time for the trajectory.

        Args:
            t (float): The time for takeoff.
        """
        dt = t - self.traj_data["takeoffTime"]
        self.traj_data["takeoffTime"] = t
        self.traj_data["landingTime"] += t
        for segment in self.traj_data["points"]:
            segment[0] += dt
        self.traj_data["started"] = True
        self.evaluator = TrajEvaluator(self.traj_data)

    def evaluate(self, time: float) -> dict[str, Any]:
        """
        Overrides (implements) superclass' evaluate. If the timestamp is before or after the trajectory, returns the
        start or end of the trajectory respectively, with the derivatives set to 0.

        Args:
            time (float): The timestamp when the trajectory must be evaluated.

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        retval = {}
        if not self.traj_data["started"] or time <= self.traj_data["takeoffTime"]: # The trajectory hasn't started yet
            segment = self.traj_data["points"][0]
            retval["load_mass"] = 0.0
            retval["target_pos"] = np.array(segment[1][:3])
            retval["target_vel"] = np.zeros(3)  # We haven't started: speed should be 0.
            retval["target_acc"] = np.zeros(3)  # We haven't started: acc should be 0.
            retval["target_rpy"] = np.array([0, 0, np.deg2rad(segment[1][3])])
            retval["target_ang_vel"] = np.zeros(3)  # We haven't started: angular velocity should be 0.
        elif time > self.traj_data["landingTime"]:
            segment = self.traj_data["points"][-1]
            retval["load_mass"] = 0.0
            retval["target_pos"] = np.array(segment[1][:3])
            retval["target_vel"] = np.zeros(3)  # We've stopped: speed should be 0.
            retval["target_acc"] = np.zeros(3)  # We've stopped: acc should be 0.
            retval["target_rpy"] = np.array([0, 0, np.deg2rad(segment[1][3])])
            retval["target_ang_vel"] = np.zeros(3)  # We've stopped: angular velocity should be 0.
        else:
            curve = self.evaluator.select_curve(time)
            retval["load_mass"] = 0.0
            retval["target_pos"] = np.array([curve.x(time), curve.y(time), curve.z(time)])
            retval["target_vel"] = np.array([curve.x(time, nu=1), curve.y(time, nu=1), curve.z(time, nu=1)])
            retval["target_acc"] = np.array([curve.x(time, nu=2), curve.y(time, nu=2), curve.z(time, nu=2)])
            retval["target_rpy"] = np.array([0, 0, np.deg2rad(curve.yaw(time))])
            retval["target_ang_vel"] = np.array([0, 0, np.deg2rad(curve.yaw(time, nu=1))])
        return retval

def extract_trajectories(skyc_file: str) -> list[SkycTrajectory]:
    """
    Creates a list of SkycTrajectories from a skyc file.

    Args:
        skyc_file (str): The string of the skyc file's path.

    Returns:
        list[SkycTrajectory]: A list with one SkycTrajectory per drone in the skyc file.
    """
    traj_data: list[dict[str, Any]] = skyc_inspector.get_traj_data(skyc_file)
    return [SkycTrajectory(t) for t in traj_data]
