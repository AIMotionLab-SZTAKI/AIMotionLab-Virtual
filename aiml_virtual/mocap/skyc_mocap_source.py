# TODO: COMMENTS, DOCSTRINGS

import numpy as np
from typing import Callable
import platform
if platform.system() == 'Windows':
    import win_precise_time as time
else:
    import time

from aiml_virtual.mocap import mocap_source
from aiml_virtual.trajectory.skyc_trajectory import SkycTrajectory
from aiml_virtual.utils.utils_general import quaternion_from_euler


class SkycMocapSource(mocap_source.MocapSource):
    """
    Todo: Docstring
    """
    def __init__(self, trajectories: list[SkycTrajectory], t: Callable[[], float], fps: float = 100):
        super().__init__()
        self.trajectories: list[SkycTrajectory] = trajectories
        self.t = t
        self.fps = fps
        self.start_mocap_thread()

    def t(self):
        raise NotImplementedError


    def mocap(self) -> None:
        while True:
            with self.lock:
                for i, traj in enumerate(self.trajectories):
                    pos = traj.evaluate(self.t())["target_pos"]
                    rpy = traj.evaluate(self.t())["target_rpy"]
                    quat = np.array(quaternion_from_euler(*rpy))
                    self._data[f"cf{i}"] = (pos, quat)
            time.sleep(1 / self.fps)




