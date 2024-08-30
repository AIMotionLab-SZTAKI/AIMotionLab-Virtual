"""
This module contains a dummy trajectory class that one can use to model more complex trajectories after.
"""

import numpy as np
from typing import Any

from aiml_virtual.trajectory import trajectory


class DummyDroneTrajectory(trajectory.Trajectory):
    """
    Dummy trajectory specifically for drones: all it does is hover at the given position at 0 speed/acceleration.
    """
    def __init__(self, target_pos: np.ndarray = np.array([0, 0, 1])):
        super().__init__()
        self.target_pos: np.ndarray = target_pos  #: The position to hold.


    def evaluate(self, time: float) -> dict[str, Any]:
        """
        Overrides (implements) superclass' evaluate, ensuring DummyDroneTrajectory is a concrete class. Returns
        a setpoint that is always the same hover setpoint regardless of time.

        Args:
            time (float): The time at which to evaluate (although here it doesn't matter).

        Returns:
            dict[str, Any]: The desired setpoint at the provided timestamp.
        """
        output = {"load_mass": 0.0,
                  "target_pos": self.target_pos,  # hold this position, no matter the time
                  "target_rpy": np.zeros(3),  # hold level, face the X axis
                  "target_vel": np.zeros(3),  # maintain 0 speed
                  "target_acc": np.zeros(3),  # maintain 0 acceleration
                  "target_ang_vel": np.zeros(3)}  # have no angular velocity
        return output