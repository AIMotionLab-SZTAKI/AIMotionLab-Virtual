import math
import numpy as np

# TODO: DOCSTRINGS
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

def quat_quat_array_multiply(quat, quat_array):

    w0, x0, y0, z0 = quat
    w1, x1, y1, z1 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)

def quat_array_quat_multiply(quat_array, quat):
    w0, x0, y0, z0 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    w1, x1, y1, z1 = quat
    return np.stack((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), axis=1)

def quat_vect_array_mult(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_quat_array_multiply(q_conjugate(q), quat_array_quat_multiply(q_array, q))[:, 1:]

def quat_vect_array_mult_passive(q, v_array):
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_array_quat_multiply(quat_quat_array_multiply(q, q_array), q_conjugate(q))[:, 1:]

def forces_from_pressures(normal, pressure, area):
    # f = np.array([0., 0., -1.])
    # F = np.dot(-normal, f) * np.outer(pressure, f) * area
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
        F = velocity * density * np.expand_dims(area, axis=1) * np.sum(velocity * (-normal), axis=1).reshape(-1, 1)
    return F

def torque_from_force(r, force):
    """by Adam Weinhardt"""
    M = np.cross(r, force)
    return M