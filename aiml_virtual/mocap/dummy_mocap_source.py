# TODO: DOCSTRINGS AND COMMENTS
import threading
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
    d_pos = np.array([np.cos(2 * np.pi * time.time() / T), np.sin(2 * np.pi * time.time() / T), 0])
    ret: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (pos, quat) in start_poses.items():
        ret[key] = (d_pos + pos, quat)
    return ret



class DummyMocapSource(mocap_source.MocapSource):
    def __init__(self, frame_generator: Callable[...,dict[str, tuple[np.ndarray, np.ndarray]]], fps: float = 120):
        super().__init__()
        self.fps: float = fps
        self.generate_frame = frame_generator
        self.start_mocap_thread()

    def generate_frame(self):
        raise NotImplementedError

    def mocap(self) -> None:
        while True:
            with self.lock:
                self._data = copy.deepcopy(self.generate_frame())
            time.sleep(1/self.fps)





