"""
This module contains utility functions and classes for the aiml_virtual package.
"""

import numpy as np
import platform
import importlib
import pkgutil
import pathlib
import math
from typing import Union, Optional
from scipy.spatial.transform import Rotation
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

def import_submodules(package_name: str) -> None:
    """
    Recursively imports all submodules under the given package.

    Args:
        package_name (str): The root package to scan for submodules.
    """
    # Get the package object
    package = importlib.import_module(package_name)

    # Get the package path as a Path object
    package_path = pathlib.Path(package.__file__).parent

    # Iterate through the package's submodules
    for (module_finder, name, ispkg) in pkgutil.walk_packages([str(package_path)]):
        full_module_name = f"{package_name}.{name}"

        if ispkg:
            # If it’s a package, recursively import its submodules
            import_submodules(full_module_name)
        else:
            # Import the module
            importlib.import_module(full_module_name)

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

def euler_from_quaternion(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).

    Args:
        w (float): quaternion w
        x (float): quaternion x
        y (float): quaternion y
        z (float): quaternion z
    Returns:
        tuple[float, float, float]: roll(x), pitch(y), yaw(z), in radians, counterclockwise
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def clamp(value: Union[float, int], bound: Union[int, float, list, tuple, np.ndarray]) -> float:
    """Helper function that clamps the given value with the specified bounds

    Args:
        value (float | int): The value to clamp
        bound Union[int, float, list, tuple, np.ndarray]: If int | float the function constrains the value into [-bound,bound]
                                                          If tuple| list | np.ndarray the value is constained into the range of [bound[0],bound[1]]

    Returns:
        float: The clamped value
    """
    if isinstance(bound, int) or isinstance(bound, float):
        if value < -bound:
            return float(-bound)
        elif value > bound:
            return float(bound)
        return float(value)
    elif isinstance(bound, tuple) or isinstance(bound, list) or isinstance(bound, np.ndarray):
        if value < bound[0]:
            return float(bound[0])
        elif value > bound[1]:
            return float(bound[1])
        return float(value)

def normalize(angle: float) -> float:
    """Normalizes the given angle into the [-pi/2, pi/2] range

    Args:
        angle (float): Input angle

    Returns:
        float: Normalized angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return angle

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

def warning(text: str) -> None:
    """
    Print in scary red letters.

    Args:
        text (str): The text of the warning.
    """
    red = "\033[91m"
    reset = "\033[0m"
    formatted_text = f"WARNING: {text}"
    print(red + formatted_text + reset)

def fix_angles(quat: np.array, *, roll: Optional[float] = None, pitch: Optional[float] = None,
               yaw: Optional[float] = None) -> np.array:
    """
    Turns a quaternion into euler angles, overwrites the euler angles provided as not None, and returns a quaterion
    matching the modified angles.

    Args:
        quat (np.array): The original quaternion, ordered w-x-y-z.
        roll (Optional[float]): The fixed angle around the x-axis.
        pitch (Optional[float]): The fixed angle around the y-axis.
        yaw (Optional[float]): The fixed angle around the z-axis.

    Returns:
        np.array: The modified quaternion
    """
    euler = Rotation.from_quat(np.roll(quat, -1)).as_euler("xyz") # np.roll because scipy Rotation uses x-y-z-w
    if roll is not None:
        euler[0] = roll
    if pitch is not None:
        euler[1] = pitch
    if yaw is not None:
        euler[2] = yaw
    new_quat = Rotation.from_euler("xyz", euler).as_quat()
    return np.roll(new_quat, 1)

def offset_angles(quat: np.array, *, roll: Optional[float] = None, pitch: Optional[float] = None,
               yaw: Optional[float] = None) -> np.array:
    """
    Turns a quaternion into euler angles, adds the provided angles to the original, and returns a quaterion
    matching the modified angles.

    Args:
        quat (np.array): The original quaternion, ordered w-x-y-z.
        roll (Optional[float]): The additional angle around the x-axis.
        pitch (Optional[float]): The additional angle around the y-axis.
        yaw (Optional[float]): The additional angle around the z-axis.

    Returns:
        np.array: The modified quaternion
    """
    euler = Rotation.from_quat(np.roll(quat, -1)).as_euler("xyz")  # np.roll because scipy Rotation uses x-y-z-w
    if roll is not None:
        euler[0] += roll
    if pitch is not None:
        euler[1] += pitch
    if yaw is not None:
        euler[2] += yaw
    new_quat = Rotation.from_euler("xyz", euler).as_quat()
    return np.roll(new_quat, 1)