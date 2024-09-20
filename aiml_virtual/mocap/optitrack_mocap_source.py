"""
This module contains a motion capture class to encapsulate the optitrack system.
"""

import copy
import sys
if sys.platform != "linux":
    class motioncapture:
        MotionCaptureOptitrack = None
else:
    import motioncapture
import numpy as np

from aiml_virtual.mocap import mocap_source


class OptitrackMocapSource(mocap_source.MocapSource):
    """
    Motion Capture source that takes its frames from the Optitrack system.
    """
    def __init__(self, ip: str = "192.168.2.141"):
        super().__init__()
        self.optitrack: motioncapture.MotionCaptureOptitrack = motioncapture.MotionCaptureOptitrack(ip)  #: motion capture handler
        self.start_mocap_thread()

    def mocap(self) -> None:
        """
        Extracts the optitrack frames using optitrack.waitForNextFrame(), and packs them into its own dictionary in
        an infnite loop. By nature this function's loop will spin at the optitrack's frequency (120Hz).
        """
        while True:
            self.optitrack.waitForNextFrame()
            data_dict = {}
            for name, obj in self.optitrack.rigidBodies.items():
                pos = obj.position
                quat = obj.rotation
                data_dict[name] = (pos, np.array([quat.x, quat.y, quat.z, quat.w]))
            with self.lock:  # use a deep copy just in case
                self._data = copy.deepcopy(data_dict)

