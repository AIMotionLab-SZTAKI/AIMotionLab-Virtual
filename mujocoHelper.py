import mujoco
import math

import numpy as np
from collections import deque

class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")


class LiveLFilter(LiveFilter):
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y

def update_drone(data, droneID, position, orientation):
    """
    Update the position and orientation of a drone
    first drone's position is data.qpos[:3], orientation is data.qpos[3:7]
    second drone's position is data.qpos[7:10], orientation is data.qpos[10:14]
    and so on
    droneID should be 0 for the first drone defined in the xml, 1 for the second etc.
    """
    startIdx = droneID * 7
    if startIdx + 6 >= data.qpos.size:
        print("Drone id: " + str(droneID) + " out of bounds of data.qpos. (Not enough drones defined in the xml)")
        return
    data.qpos[startIdx:startIdx + 3] = position
    data.qpos[startIdx + 3:startIdx + 7] = orientation


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
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

        

def update_follow_cam(qpos, droneID, cam, azim_filter_sin=None, azim_filter_cos=None, elev_filter_sin=None, elev_filter_cos=None):
    """
    Update the position and orientation of the camera that follows the drone from behind
    qpos is the array in which the position and orientation of all the drones are stored

    Smoothing the 2 angle signals with 4 low-pass filters
    """
    MAX_CHANGE = 3

    startIdx = droneID * 7
    position = qpos[startIdx:startIdx + 3]
    orientation = qpos[startIdx + 3:startIdx + 7]

    roll_x, pitch_y, yaw_z = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])

    new_azim = -math.degrees(roll_x)
    new_elev = -math.degrees(pitch_y) - 20

    cam.lookat = position

    if azim_filter_sin and azim_filter_cos:
        cosa = math.cos(math.radians(new_azim))
        sina = math.sin(math.radians(new_azim))

        cosa = azim_filter_cos(cosa)
        sina = azim_filter_sin(sina)

        cam.azimuth = math.degrees(math.atan2(sina, cosa))

    if elev_filter_sin and elev_filter_cos:
        cosa = math.cos(math.radians(new_elev))
        sina = math.sin(math.radians(new_elev))

        cosa = elev_filter_cos(cosa)
        sina = elev_filter_sin(sina)

        cam.elevation = math.degrees(math.atan2(sina, cosa))

    else:
        cam.azimuth = new_azim
        cam.elevation = new_elev
