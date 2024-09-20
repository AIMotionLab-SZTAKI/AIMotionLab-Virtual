"""
This module contains a dummy motion capture class which can be used to simulate mocap data for objects, as well as
some example frame generators.
"""

import platform
import copy
from typing import Callable
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

import numpy as np

from aiml_virtual.mocap import mocap_source

def generate_circular_paths(start_poses: dict[str, tuple[np.ndarray, np.ndarray]], T:float = 5) -> \
        dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Generates a motion capture frame where the output objects are the same as the input objects, but their position is
    moved in a circle around the given start pose.

    Args:
        start_poses (dict[str, tuple[np.ndarray, np.ndarray]]): The starting dictionary, which specifies the objects
            to include in the frame, and their starting pose (which will be offset with a circle).
        T (float): The period of the circle the objects will travel.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: A motion capture frame (with the same elements as the input).
    """
    d_pos = np.array([np.cos(2 * np.pi * time.time() / T), np.sin(2 * np.pi * time.time() / T), 0]) # position offset
    ret: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (pos, quat) in start_poses.items():
        ret[key] = (d_pos + pos, quat)
    return ret


class DummyMocapSource(mocap_source.MocapSource):
    """
    Class that emulates a motion capture system, but with specified objects in its frame. It may be initialized with
    a frame rate and a callable that must take no arguments and return a dict[str, tuple[np.ndarray, np.ndarray]]].
    The callable will be  invoked (approximately) at the given rate to generate motion capture frames.

    .. note::
        The mocap source will have a separate thread associated with it, where the frame generator function provided
        will be called in an infinite loop. However, the frame generator function must take no arguments. If one still
        wants to provide arguments to the function, the arguments should be specified using functools.partial like so:

        .. code-block:: python

            mocap_frame_generator = functools.partial(function_with_arguments, arg1=something, arg2=something_else)

    """
    def __init__(self, frame_generator: Callable[[],dict[str, tuple[np.ndarray, np.ndarray]]], fps: float = 120):
        super().__init__()
        self.fps: float = fps  #: The rate at which the frame generator gets called.
        self.generate_frame: Callable[[],dict[str, tuple[np.ndarray, np.ndarray]]] = frame_generator  #: The function that will be used to generate mocap frames.
        self.start_mocap_thread()


    def mocap(self) -> None:
        """
        Calls self.generate_frame (which was the frame generator provided in the constructor) in an infinite loop.
        """
        while True:
            with self.lock:
                self._data = copy.deepcopy(self.generate_frame())
            time.sleep(1/self.fps)





