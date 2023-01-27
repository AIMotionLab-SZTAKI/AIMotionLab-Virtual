import mujoco
import math

import numpy as np
from collections import deque


class LiveFilter:
    """Base class for live filters.
       from https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/
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
           from https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/

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
        #print("_xs: " + str(self._xs))
        #print("_ys: " + str(self._ys))
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y


def update_drone(data, droneID, position, orientation):
    """
    Old, do not use this anymore!
    Update the position and orientation of a drone
    first drone's position is data.qpos[:3], orientation is data.qpos[3:7]
    second drone's position is data.qpos[7:10], orientation is data.qpos[10:14]
    and so on
    droneID should be 0 for the first drone defined in the xml, 1 for the second etc.
    """
    startIdx = droneID * 7
    if startIdx + 6 >= data.qpos.size:
        print("Drone id: " + str(droneID) +
              " out of bounds of data.qpos. (Not enough drones defined in the xml)")
        return
    data.qpos[startIdx:startIdx + 3] = position
    data.qpos[startIdx + 3:startIdx + 7] = orientation


def get_joint_name_list(mjmodel: mujoco.MjModel):
    """
    Create a list of valid joint names of a mujoco model
    """
    n = mjmodel.njnt
    name_list = []
    for i in range(n):
        o_name = mjmodel.joint(i).name
        if o_name != "":
            name_list.append(o_name)

    return name_list


def get_geom_name_list(mjmodel: mujoco.MjModel):
    """
    Create a list of valid geom names of a mujoco model
    """
    n = mjmodel.ngeom
    name_list = []
    for i in range(n):
        o_name = mjmodel.geom(i).name
        if o_name != "":
            name_list.append(o_name)

    return name_list


def get_body_name_list(mjmodel: mujoco.MjModel):
    """
    Create a list of valid body names of a mujoco model
    """
    n = mjmodel.nbody
    name_list = []
    for i in range(n):
        o_name = mjmodel.body(i).name
        if o_name != "":
            name_list.append(o_name)

    return name_list


def euler_from_quaternion(w, x, y, z):
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

    return [roll_x, pitch_y, yaw_z]  # in radians


def quaternion_from_euler(roll, pitch, yaw):
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


def quaternion_multiply(quaternion0, quaternion1):
    """Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((
                    -x1*x0 - y1*y0 - z1*z0 + w1*w0,
                    x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                    x1*y0 - y1*x0 + z1*w0 + w1*z0), dtype=np.float64)

def q_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def qv_mult(q1, v1):
    """For active rotation. If passive rotation is needed, use q1 * q2 * q1^(-1)"""
    q2 = np.append(0.0, v1)
    return quaternion_multiply(q_conjugate(q1), quaternion_multiply(q2, q1))[1:]


def euler_rad_to_euler_deg(array_3elem):
    return [math.degrees(array_3elem[0]), math.degrees(array_3elem[1]), math.degrees(array_3elem[2])]


def update_onboard_cam(drone_qpos, cam, azim_filter_sin=None, azim_filter_cos=None, elev_filter_sin=None, elev_filter_cos=None, elev_offs=30):
    """
    Update the position and orientation of the camera that follows the drone from behind
    qpos is the array in which the position and orientation of all the drones are stored

    Smoothing the 2 angle signals with 4 low-pass filters so that the camera would not shake.
    It's not enough to only filter the angle signal, because when the drone turns,
    the angle might jump from 180 degrees to -180 degrees, and the filter tries to smooth out
    the jump (the camera ends up turning a 360). Instead, take the sine and the cosine of the
    angle, filter them, and convert them back with atan2().
    """
    MAX_CHANGE = 3
    position = drone_qpos[0:3]
    orientation = drone_qpos[3:7]

    roll_x, pitch_y, yaw_z = euler_from_quaternion(
        orientation[0], orientation[1], orientation[2], orientation[3])

    new_azim = math.degrees(yaw_z)
    new_elev = -math.degrees(pitch_y) - elev_offs

    cam.lookat = position

    if azim_filter_sin and azim_filter_cos:
        cosa = math.cos(math.radians(new_azim))
        sina = math.sin(math.radians(new_azim))

        cosa = azim_filter_cos(cosa)
        sina = azim_filter_sin(sina)

        cam.azimuth = math.degrees(math.atan2(sina, cosa))

    else:
        cam.azimuth = new_azim

    if elev_filter_sin and elev_filter_cos:
        cosa = math.cos(math.radians(new_elev))
        sina = math.sin(math.radians(new_elev))

        cosa = elev_filter_cos(cosa)
        sina = elev_filter_sin(sina)

        cam.elevation = math.degrees(math.atan2(sina, cosa))

    else:
        cam.elevation = new_elev
