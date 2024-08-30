"""
This module contains utility functions and classes for the aimotion_virtual paclage.
"""

import numpy as np
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> list[float]:
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

class PausableTime:
    """
    Class that mimics the behaviour of time.time(), but shows the time since it was last reset (or initialized), and
    it can be paused and resumed like a stopwatch.
    """
    def __init__(self):
        self.last_started: float = time.time()  #: When resume was last called.
        self.sum_time: float = 0.0  #: The amount of time the clock ran the last time it was paused.
        self.ticking: bool = True  #: Whether the clock is ticking.

    def pause(self) -> None:
        """
        Pauses the timer, which freezes the time it displays to its current state.
        """
        if self.ticking:
            self.sum_time += time.time() - self.last_started
            self.ticking = False

    def resume(self) -> None:
        """
        Resumes time measurement, which starts updating the displayed time again.
        """
        if not self.ticking:
            self.last_started = time.time()
            self.ticking = True

    def __call__(self, *args, **kwargs) -> float:
        """
        Shows the cummulated time the clock was running since it was last reset.

        Returns:
            float: The cummulated measured time.
        """
        if self.ticking:
            return self.sum_time + (time.time() - self.last_started)
        else:
            return self.sum_time

    def reset(self) -> None:
        """
        Resets the measured time to 0, and starts the clock ticking if it was paused.

        .. note::
            This is the same as __init__, but in theory it's not good practice to call __init__ by hand.
        """
        self.last_started: float = time.time()
        self.sum_time = 0.0
        self.ticking: bool = True