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
    def __init__(self, trajectories: list[SkycTrajectory], t: Callable[[], float], fps: float = 50):
        super().__init__()
        self.trajectories: dict[str, SkycTrajectory] = {f"vcf{i}": v for i, v in enumerate(trajectories)}
        self.t = t
        self.fps = fps
        self.start_mocap_thread()

    def t(self):
        raise NotImplementedError


    def mocap(self) -> None:
        while True:
            with self.lock:
                for k, v in self.trajectories.items():
                    pos = v.evaluate(self.t())["target_pos"]
                    rpy = v.evaluate(self.t())["target_rpy"]
                    quat = np.array(quaternion_from_euler(*rpy))
                    self._data[k] = (pos, quat)
            time.sleep(1 / self.fps)




