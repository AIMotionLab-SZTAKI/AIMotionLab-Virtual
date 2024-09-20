# TODO: DOCSTRINGS AND COMMENTS
import threading
import platform
import copy
from typing import Callable
import motioncapture
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

import numpy as np

from aiml_virtual.mocap import mocap_source


class OptitrackMocapSource(mocap_source.MocapSource):
    def __init__(self, ip: str = "192.168.2.141"):
        super().__init__()
        self.optitrack: motioncapture.MotionCaptureOptitrack = motioncapture.MotionCaptureOptitrack(ip)
        self.start_mocap_thread()

    def mocap(self) -> None:
        while True:
            self.optitrack.waitForNextFrame()
            data_dict = {}
            for name, obj in self.optitrack.rigidBodies.items():
                pos = obj.position
                quat = obj.rotation
                data_dict[name] = (pos, np.array([quat.x, quat.y, quat.z, quat.w]))
            with self.lock:
                self._data = copy.deepcopy(data_dict)

