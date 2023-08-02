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


def skipper(fname):
    """
    Skips the header of 
    """
    with open(fname) as fin:
        no_comments = (line for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None) # skip header
        for row in no_comments:
            yield row


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

def get_freejoint_name_list(mjmodel: mujoco.MjModel):
    """
    Create a list of valid free joint names of a mujoco model
    """
    n = mjmodel.njnt
    name_list = []
    for i in range(n):
        joint = mjmodel.joint(i)
        if joint.name != "" and joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            name_list.append(joint.name)

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

def get_mocapbody_name_list(mjmodel: mujoco.MjModel):
    """
    Create a list of valid mocap body names of a mujoco model
    """
    n = mjmodel.nbody
    name_list = []
    for i in range(n):
        body = mjmodel.body(i)
        body_name = body.name
        if body_name != "" and body.mocapid[0] > -1:
            name_list.append(body_name)

    return name_list

def length(vector):

    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

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

def q_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]


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


def qv_mult(q1, v1):
    """For active rotation. If passive rotation is needed, use q1 * q2 * q1^(-1)"""
    q2 = np.append(0.0, v1)
    return quaternion_multiply(q_conjugate(q1), quaternion_multiply(q2, q1))[1:]

def qv_mult_passive(q1, v1):
    q2 = np.append(0.0, v1)
    return quaternion_multiply(quaternion_multiply(q1, q2), q_conjugate(q1))[1:]



def quat_array_conjugate(quat_array):
    quat_array[:, 1] *= -1
    quat_array[:, 2] *= -1
    quat_array[:, 3] *= -1
    return quat_array

def quat_array_quat_multiply(quat_array, quat):
    
    w0, x0, y0, z0 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    w1, x1, y1, z1 = quat
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)

def quat_quat_array_multiply(quat, quat_array):

    w0, x0, y0, z0 = quat
    w1, x1, y1, z1 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)


def quat_array_quat_array_multiply(quat_array0, quat_array1):

    w0, x0, y0, z0 = quat_array0[:, 0], quat_array0[:, 1], quat_array0[:, 2], quat_array0[:, 3]
    w1, x1, y1, z1 = quat_array1[:, 0], quat_array1[:, 1], quat_array1[:, 2], quat_array1[:, 3]
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)

def quat_vect_array_mult(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_quat_array_multiply(q_conjugate(q), quat_array_quat_multiply(q_array, q))[:, 1:]


def quat_vect_array_mult_passive(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_array_quat_multiply(quat_quat_array_multiply(q, q_array), q_conjugate(q))[:, 1:]


def euler_rad_to_euler_deg(array_3elem):
    return [math.degrees(array_3elem[0]), math.degrees(array_3elem[1]), math.degrees(array_3elem[2])]


def force_from_pressure(normal, pressure, area):
    """by Adam Weinhardt"""
    f = np.array([0., 0., -1.])
    F = np.dot(-normal, f) * f * pressure * area
    return F

def torque_from_force(r, force):
    """by Adam Weinhardt"""
    M = np.cross(r, force)
    return M

def forces_from_pressures(normal, pressure, area):
    
    #f = np.array([0., 0., -1.])
    #F = np.dot(-normal, f) * np.outer(pressure, f) * area
    F = np.outer(pressure, -normal) * area
    return F

def forces_from_velocities(normal, velocity, area):
    density = 1.1839 #kg/m^3
    F = velocity * density * area * np.dot(velocity, -normal).reshape(-1, 1)
    return F

def update_onboard_cam(qpos, cam, azim_filter_sin=None, azim_filter_cos=None, elev_filter_sin=None, elev_filter_cos=None, elev_offs=30):
    """
    Update the position and orientation of the camera that follows the vehicle from behind
    qpos is the array in which the position and orientation of all the vehicles are stored

    Smoothing the 2 angle signals with 4 low-pass filters so that the camera would not shake.
    It's not enough to only filter the angle signal, because when the vehicle turns,
    the angle might jump from 180 degrees to -180 degrees, and the filter tries to smooth out
    the jump (the camera ends up turning a 360). Instead, take the sine and the cosine of the
    angle, filter them, and convert them back with atan2().
    """
    MAX_CHANGE = 3
    position = qpos[0:3]
    orientation = qpos[3:7]

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
