import mujoco
import math
import os
from python_utils import delta_to_seconds
from stl import mesh
import matplotlib.pyplot as plt

import numpy as np
from collections import deque
from sympy import diff, sin, acos, lambdify
from sympy.abc import x as sympyx
from PIL import Image
from numba import njit

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

@njit
def q_conjugate_opt(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply_opt(q0, q1):
    w0, x0, y0, z0 = np.split(q0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    return np.concatenate([-x1*x0 - y1*y0 - z1*z0 + w1*w0, x1*w0 + y1*z0 - z1*y0 + w1*x0, -x1*z0 + y1*w0 + z1*x0 + w1*y0, x1*y0 - y1*x0 + z1*w0 + w1*z0], axis=-1)

def qv_mult_opt(q, v):
    q2 = np.concatenate([np.zeros((v.shape[0], 1)), v], axis=-1)
    res = quaternion_multiply_opt(q_conjugate_opt(q), quaternion_multiply_opt(q2, q))[:, 1:]
    return res

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

def distance(pts1, pts2):
    return np.sqrt(np.sum((pts1-pts2)**2, axis=0))

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
    if normal.ndim == 1:
        F = np.outer(pressure, -normal) * area
    else:
        F = np.expand_dims(pressure, axis=1) * (-normal) * np.expand_dims(area, axis=1)
    return F

def forces_from_velocities(normal, velocity, area):
    density = 1.293 #kg/m^3
    if normal.ndim == 1:
        F = velocity * density * area * np.dot(velocity, -normal).reshape(-1, 1)
    else:
        F = velocity * density * area * np.sum(velocity * (-normal), axis=1).reshape(-1, 1)
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
    new_elev = math.degrees(pitch_y) - elev_offs

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
    

def move_point_on_sphere(point, delta_theta, delta_phi):
    # convert to polar coordinates
    r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    theta = math.acos(point[2] / r)
    phi = math.atan2(point[1], point[0])

    # calculate new angles
    theta_n = theta + delta_theta
    phi_n = phi + delta_phi

    # convert back to cartesian
    x_n = r * math.sin(theta_n) * math.cos(phi_n)
    y_n = r * math.sin(theta_n) * math.sin(phi_n)
    z_n = r * math.cos(theta_n)

    return np.array((x_n, y_n, z_n))

def move_points_on_sphere(points, delta_theta, delta_phi):
    # convert to polar coordinates
    r = np.sqrt(points[:, :, 0]**2 + points[:, :, 1]**2 + points[:, :, 2]**2)
    theta = np.arccos(points[:, :, 2] / r)
    phi = np.arctan2(points[:, :, 1], points[:, :, 0])

    # calculate new angles
    theta_n = theta + delta_theta
    phi_n = phi + delta_phi

    # convert back to cartesian
    x_n = r * np.sin(theta_n) * np.cos(phi_n)
    y_n = r * np.sin(theta_n) * np.sin(phi_n)
    z_n = r * np.cos(theta_n)

    return np.dstack((x_n, y_n, z_n))

def teardrop_curve(a=5., exp=1.3, resolution=100, height_scale=1.0, sampling="curv"):

    if sampling == "curv":
        # create sampling points based on the curvature of the function
        xs = curv_space(a, exp, height_scale, resolution)
    elif sampling == "lin":
        xs = np.linspace(0., 2 * a, resolution)
    else:
        raise RuntimeError("Unknown sampling type")
    
    #xs[xs > 2 * a] = 2 * a

    zps = height_scale * a * np.sin(np.arccos((-xs + a) / a)) * (np.sin(np.arccos((-xs + a) / a) / 2.))**exp

    return xs, zps



def create_teardrop_points(a=5., exp=1.3, resolution=100, height_scale=1.0, tilt=0.0, sampling="curv"):

    xs, zps = teardrop_curve(a, exp, resolution, height_scale, sampling)
    ys = np.zeros(2 * resolution - 2)
    zns = -zps[1:-1].copy()

    zs = np.append(zps, np.flip(zns))
    xs = np.append(xs, np.flip(xs[1:-1]))

    points = np.vstack((xs, ys, zs)).T

    euler_tilt = np.array((0.0, tilt, 0.0))

    quat_tilt = quaternion_from_euler(*euler_tilt)

    points = quat_vect_array_mult(quat_tilt, points)
    
    #ax = plt.axes(projection="3d")
    #ax.plot(xs, ys, zs)
    #ax.plot(points[:, 0], points[:, 1], points[:, 2])

    #plt.plot(xs, zs)
    #plt.show()

    return points



def create_radar_field_stl(a=5., exp=1.3, rot_resolution=90, resolution=100, height_scale=1.0, tilt=0.0, filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                           sampling="curv"):

    
    points = create_teardrop_points(a, exp, resolution, height_scale, tilt, sampling)


    rotated_lobes = [points]

    rot_step = 2 * math.pi / rot_resolution
    current_rotation = rot_step

    for i in range(rot_resolution - 1):

        euler = np.array((0.0, 0.0, current_rotation))

        quat = quaternion_from_euler(*euler)

        rot_points = quat_vect_array_mult(quat, points)
        #ax.plot(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2])

        rotated_lobes += [rot_points]

        current_rotation += rot_step

    #plt.show()

    triangles = []

    for i in range(rot_resolution):

        for j in range(2 * resolution - 2):

            if j == 0:
                #triangles += [rotated_lobes[0][0], rotated_lobes[i][1], rotated_lobes[i + 1][0]]
                if i == rot_resolution - 1:
                    triangles += [rotated_lobes[0][0]]
                    triangles += [rotated_lobes[i][1]]
                    triangles += [rotated_lobes[0][1]]
                elif i < rot_resolution - 1:
                    triangles += [rotated_lobes[0][0]]
                    triangles += [rotated_lobes[i][1]]
                    triangles += [rotated_lobes[i + 1][1]]
                
            
            else:
                if j == 2 * resolution - 3:
                    if i == rot_resolution - 1:
                        triangles += [rotated_lobes[i][j]]
                        triangles += [rotated_lobes[0][0]]
                        triangles += [rotated_lobes[0][j]]
                    elif i < rot_resolution - 1:
                        triangles += [rotated_lobes[i][j]]
                        triangles += [rotated_lobes[0][0]]
                        triangles += [rotated_lobes[i + 1][j]]

                
                else:
                    if i == rot_resolution - 1:
                        triangles += [rotated_lobes[i][j]]
                        triangles += [rotated_lobes[i][j + 1]]
                        triangles += [rotated_lobes[0][j]]

                        triangles += [rotated_lobes[i][j + 1]]
                        triangles += [rotated_lobes[0][j + 1]]
                        triangles += [rotated_lobes[0][j]]

                    else:
                        
                        triangles += [rotated_lobes[i][j]]
                        triangles += [rotated_lobes[i][j + 1]]
                        triangles += [rotated_lobes[i + 1][j]]

                        triangles += [rotated_lobes[i][j + 1]]
                        triangles += [rotated_lobes[i + 1][j + 1]]
                        triangles += [rotated_lobes[i + 1][j]]
                        

    triangles = np.array(triangles)
    #print(triangles.shape)

    #ax = plt.axes(projection="3d")

    #plot_from = 0
    #plot_to = triangles.shape[0]
    #ax.plot(triangles[plot_from:plot_to, 0], triangles[plot_from:plot_to, 1], triangles[plot_from:plot_to, 2])
    #plt.show()

    num_triangles = int(triangles.shape[0] / 3)
    radar_field_mesh = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))

    v_idx = 0
    for i in range(num_triangles):
        for j in range(3):
            radar_field_mesh.vectors[i][j] = triangles[v_idx]
            v_idx += 1
    
    filename = "radar_field_a" + str(a) + "_exp" + str(exp) + "_rres" + str(rot_resolution) + "_res" + str(resolution) +\
               "_hs" + str(height_scale) + "_tilt" + str(tilt) + "_" + sampling + ".stl"

    radar_field_mesh.save(os.path.join(filepath, filename))

    print("[mujoco_helper] Saved radar mesh at: " + os.path.normpath(os.path.join(filepath, filename)))

    return filename


def create_teardrop_stl(a=5., exp=1.3, rot_resolution=90, resolution=100, height_scale=1.0, tilt=0.0, filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
                           sampling="curv"):
    
    xs, zps = teardrop_curve(a, exp, resolution, height_scale, sampling)
    ys = np.zeros(resolution)

    points = np.vstack((xs, ys, zps)).T

    points[-1][2] = 0.0

    points_list = []

    rot_step = 2 * math.pi / rot_resolution

    current_rotation = 0

    for i in range(rot_resolution):
        euler = np.array((current_rotation, tilt, 0.0))
        quat = quaternion_from_euler(*euler)
        rot_points = quat_vect_array_mult(quat, points)

        points_list += [rot_points]
        current_rotation += rot_step

    triangles = []

    for i in range(rot_resolution):
        for j in range(resolution - 1):
            if j == 0:
                if i == rot_resolution - 1:
                    triangles += [points_list[0][0]]
                    triangles += [points_list[i][1]]
                    triangles += [points_list[0][1]]
                
                else:
                    triangles += [points_list[0][0]]
                    triangles += [points_list[i][1]]
                    triangles += [points_list[i + 1][1]]
            
            else:
                if i == rot_resolution - 1:
                    triangles += [points_list[i][j]]
                    triangles += [points_list[i][j + 1]]
                    triangles += [points_list[0][j]]

                    triangles += [points_list[i][j + 1]]
                    triangles += [points_list[0][j + 1]]
                    triangles += [points_list[0][j]]

                else:
                    triangles += [points_list[i][j]]
                    triangles += [points_list[i][j + 1]]
                    triangles += [points_list[i + 1][j]]

                    triangles += [points_list[i][j + 1]]
                    triangles += [points_list[i + 1][j + 1]]
                    triangles += [points_list[i + 1][j]]

    
    triangles = np.array(triangles)
    num_triangles = int(triangles.shape[0] / 3)
    teardrop_mesh = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))

    v_idx = 0
    for i in range(num_triangles):
        for j in range(3):
            teardrop_mesh.vectors[i][j] = triangles[v_idx]
            v_idx += 1
    
    filename = "teardrop_a" + str(a) + "_exp" + str(exp) + "_rres" + str(rot_resolution) + "_res" + str(resolution) + "_hs" + str(height_scale) + "_tilt" + "_" + sampling + ".stl"

    teardrop_mesh.save(os.path.join(filepath, filename))
    print("[mujoco_helper] Saved teardrop mesh at: " + os.path.normpath(os.path.join(filepath, filename)))
    
    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(projection='3d')
    #for pts in points_list:
    #    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    #
    #ax.axis("equal")
    #plt.show()

    return filename


def clamp(value, min, max):

    if value <= min:
        return min
    if value >= max:
        return max
    
    return value

def curv_space(a, exp, height_scale, num_samples) -> np.array:
    """
    This function creates sampling points based on the curvature of the function.
    The greater the curvature, the more sampling points are created.
    """

    upscale = 2

    squeeze = 0.0001

    xs_lin = np.linspace(squeeze, 2 * a - squeeze, num_samples * upscale)
    #zs = a * np.sin(np.arccos((-xs_lin + a) / a)) * (np.sin(np.arccos((-xs_lin + a) / a) / 2.))**exp

    expr = height_scale * a * sin(acos((a - sympyx) / a)) * sin(acos((a - sympyx) / a) / 2)**exp
    d_expr = diff(expr)
    dd_expr = diff(d_expr)

    f = lambdify(sympyx, expr, "numpy")
    f_bar = lambdify(sympyx, d_expr, "numpy")
    f_barbar = lambdify(sympyx, dd_expr, "numpy")

    zs = f(xs_lin)
    z_diff = f_bar(xs_lin)
    z_diffdiff = f_barbar(xs_lin)

    xs = np.empty(num_samples * upscale)


    xs[0] = 0.0

    x_step = 2 * a / ((num_samples * upscale) - 1)

    #z_diff = np.empty(num_samples)
    #z_diff[0] = 0.0
    #z_diffdiff = np.empty(num_samples)
    #z_diffdiff[0] = 0.0

    for i in range((num_samples * upscale) - 1):


        #z_diff[i + 1] = (zs[i + 1] - zs[i]) / x_step
        #z_diffdiff[i + 1] = (z_diff[i + 1] - z_diff[i]) / x_step

        #k_inv = clamp((1.0 + z_diff[i + 1]**2)**1.5 / abs(z_diffdiff[i + 1]) / 100., 5.0, 100.)
        k_inv = (1.0 + z_diff[i + 1]**2)**1.5 / abs(z_diffdiff[i + 1])

        #print(k_inv)

        #xs[i + 1] = xs[i] + (k_inv / clamp(abs(z_diff[i + 1]), 0.4, 2.5))
        xs[i + 1] = xs[i] + (clamp(k_inv, 0.5 * a, 5. * a) / clamp(abs(z_diff[i + 1]), 0.4, 2.5))

    #print(xs)

    #plt.plot(xs_lin, z_diff)
    xs = xs[0::upscale]

    corrector = xs[-1] / xs_lin[-1]

    xs /= corrector

    #print(xs)

    return xs


def radars_see_point(radars, point):

    if radars is None:
        return False

    for radar in radars:
        if radar.sees_point(point):
            return True

    return False


def radars_see_points(radars, points):

    if radars is None:
        return False

    bool_arr = np.zeros((points.shape[0], points.shape[1]), dtype=bool)

    for radar in radars:
        bool_arr = np.logical_or(bool_arr, radar.sees_points(points))

    return bool_arr

def create_2D_slice(slice_height, terrain_hfield, radars=None, save_folder="", save_images=False):

    dimensions = terrain_hfield.size[:3]
    x_offset, y_offset = dimensions[0], dimensions[1]
    resx, resy = terrain_hfield.ncol[0], terrain_hfield.nrow[0]

    hfield_copy = np.copy(terrain_hfield.data)
    sh_normalized = slice_height / dimensions[2]
    slice2D = hfield_copy >= sh_normalized

    if radars is not None:
        slice2D_radars = np.empty_like(terrain_hfield.data, dtype=bool)
        
        points_grid = np.empty((resy, resx, 3))

        points_grid[:, :, 0] = np.linspace(-x_offset, x_offset, resx)
        points_grid[:, :, 1] = np.linspace(-y_offset, y_offset, resy).reshape(resy, 1)
        points_grid[:, :, 2] = slice_height

        slice2D_radars = radars_see_points(radars, points_grid)
        slice2D = np.logical_or(slice2D, slice2D_radars)
        
    if save_images:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        im = Image.fromarray(np.flip(slice2D, 0) * 255)
        #im = Image.fromarray(slice2D * 255)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        if hasattr(create_2D_slice, "i"):
            im.save(os.path.join(save_folder, "slice_" + str(create_2D_slice.i) + ".png"))
        else:
            im.save(os.path.join(save_folder, "slice_h" + str(slice_height) + ".png"))

    return slice2D


def create_3D_bool_array(terrain_hfield, radars=None, save_folder="", save_images=False):

    dimx, dimy, dimz = terrain_hfield.size[0] * 2., terrain_hfield.size[1] * 2., terrain_hfield.size[2]

    resx, resy = terrain_hfield.ncol[0], terrain_hfield.nrow[0]

    pixelsize_x = dimx / resx
    pixelsize_y = dimy / resy

    if abs(pixelsize_x - pixelsize_y) > 0.0001:
        raise RuntimeError("Pixel sizes must match in x and y directions.")

    n_slices = int(round((dimz / dimx) * resx))

    height_step = dimz / (n_slices - 1)

    slices = []

    for i in range(n_slices):
        print("Computing slice at height: ", i * height_step)
        create_2D_slice.i = i
        slices += [create_2D_slice(i * height_step, terrain_hfield, radars, save_folder, save_images)]
    
    slices = np.array(slices, dtype=bool)
    #slices = np.array(slices)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    np.save(os.path.join(save_folder, "3D_bool_space.npy"), slices)

    return slices