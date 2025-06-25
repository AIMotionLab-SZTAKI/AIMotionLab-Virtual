import numpy as np

"""
Module providing quaternion-based rotation utilities and geometric computations.

This module includes functions for quaternion multiplication and conjugation, vector rotations 
using quaternions, force and torque calculations based on pressure and velocity, and triangle 
geometry helpers (normals and centers).
"""

def quaternion_multiply(quaternion0, quaternion1):
    """
    Multiply two quaternions.

    Computes the Hamilton product between `quaternion0` and `quaternion1`, returning
    a new quaternion representing the combined rotation.

    Args:
        quaternion0 (np.ndarray): First quaternion as an array [w, x, y, z].
        quaternion1 (np.ndarray): Second quaternion as an array [w, x, y, z].

    Returns:
        np.ndarray: The product quaternion [w, x, y, z] of type float64.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array((
                    -x1*x0 - y1*y0 - z1*z0 + w1*w0,
                    x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                    x1*y0 - y1*x0 + z1*w0 + w1*z0), dtype=np.float64)

def q_conjugate(q):
    """
    Compute the conjugate of a quaternion.

    The conjugate of quaternion [w, x, y, z] is [w, -x, -y, -z],
    which corresponds to the inverse rotation for unit quaternions.

    Args:
        q (np.ndarray): Quaternion as an array [w, x, y, z].

    Returns:
        np.ndarray: Conjugated quaternion [w, -x, -y, -z].
    """
    w, x, y, z = q
    return [w, -x, -y, -z]

def qv_mult(q1, v1):
    """
    Rotate a vector by a quaternion (active rotation).

    Applies the rotation defined by quaternion `q1` to vector `v1`.
    Uses the formula: v' = q1 * [0, v] * q1_conjugate.

    Args:
        q1 (np.ndarray): Rotation quaternion [w, x, y, z].
        v1 (np.ndarray): 3D vector to rotate.

    Returns:
        np.ndarray: Rotated 3D vector.
    """
    q2 = np.append(0.0, v1)
    return quaternion_multiply(q_conjugate(q1), quaternion_multiply(q2, q1))[1:]

def quat_quat_array_multiply(quat, quat_array):
    """
    Multiply a single quaternion by each quaternion in an array.

    Vectorized version of quaternion multiplication for an array of quaternions.

    Args:
        quat (np.ndarray): Single quaternion [w, x, y, z].
        quat_array (np.ndarray): Array of quaternions with shape (m, 4).

    Returns:
        np.ndarray: Array of product quaternions with shape (m, 4).
    """
    w0, x0, y0, z0 = quat
    w1, x1, y1, z1 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    return np.stack((-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0), axis=1)

def quat_array_quat_multiply(quat_array, quat):
    """
    Multiply each quaternion in an array by a single quaternion.

    Args:
        quat_array (np.ndarray): Array of quaternions with shape (m, 4).
        quat (np.ndarray): Single quaternion [w, x, y, z].

    Returns:
        np.ndarray: Array of product quaternions with shape (m, 4).
    """
    w0, x0, y0, z0 = quat_array[:, 0], quat_array[:, 1], quat_array[:, 2], quat_array[:, 3]
    w1, x1, y1, z1 = quat
    return np.stack((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), axis=1)

def quat_vect_array_mult(q, v_array):
    """
    Rotate an array of vectors by a quaternion (active rotation).

    Args:
        q (np.ndarray): Rotation quaternion [w, x, y, z].
        v_array (np.ndarray): Array of 3D vectors with shape (n, 3).

    Returns:
        np.ndarray: Rotated vectors with shape (n, 3).
    """
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_quat_array_multiply(q_conjugate(q), quat_array_quat_multiply(q_array, q))[:, 1:]

def quat_vect_array_mult_passive(q, v_array):
    """
    Rotate an array of vectors by a quaternion (passive rotation).

    Uses the formula: v' = q * [0, v] * q_conjugate.

    Args:
        q (np.ndarray): Rotation quaternion [w, x, y, z].
        v_array (np.ndarray): Array of 3D vectors with shape (n, 3).

    Returns:
        np.ndarray: Rotated vectors with shape (n, 3).
    """
    q_array = np.append(np.zeros((v_array.shape[0], 1)), v_array, axis=1)
    return quat_array_quat_multiply(quat_quat_array_multiply(q, q_array), q_conjugate(q))[:, 1:]

def forces_from_pressures(normal, pressure, area):
    """
    Compute forces from pressure applied on surfaces.

    Calculates the force vector(s) by projecting pressure along surface normals.

    Args:
        normal (np.ndarray): Surface normal as shape (3,) or (n, 3).
        pressure (np.ndarray): Pressure magnitude(s) as shape (n,) or scalar.
        area (np.ndarray): Surface area(s) as shape (n,) or scalar.

    Returns:
        np.ndarray: Force vector(s) with shape matching normals.
    """
    # f = np.array([0., 0., -1.])
    # F = np.dot(-normal, f) * np.outer(pressure, f) * area
    if normal.ndim == 1:
        F = np.outer(pressure, -normal) * area
    else:
        F = np.expand_dims(pressure, axis=1) * (-normal) * np.expand_dims(area, axis=1)
    return F

def forces_from_velocities(normal, velocity, area):
    """
    Compute aerodynamic forces from velocity on surfaces.

    Uses dynamic pressure approximation F = rho * A * (v dot n) * v,
    where rho is air density (1.293 kg/m^3).

    Args:
        normal (np.ndarray): Surface normal as shape (3,) or (n, 3).
        velocity (np.ndarray): Velocity vector(s) as shape (3,) or (n, 3).
        area (np.ndarray): Surface area(s) as shape (n,) or scalar.

    Returns:
        np.ndarray: Force vector(s) with shape matching normals.
    """
    density = 1.293 #kg/m^3
    if normal.ndim == 1:
        F = velocity * density * area * np.dot(velocity, -normal).reshape(-1, 1)
    else:
        F = velocity * density * np.expand_dims(area, axis=1) * np.sum(velocity * (-normal), axis=1).reshape(-1, 1)
    return F

def torque_from_force(r, force):
    """
    Compute torque from a force applied at a position.

    Uses the cross product M = r x F,
    where `r` is the position vector and `force` is the force vector.

    Args:
        r (np.ndarray): Position vector(s) with shape (3,) or (n, 3).
        force (np.ndarray): Force vector(s) with shape (3,) or (n, 3).

    Returns:
        np.ndarray: Torque vector(s) with same shape as inputs.
    """
    M = np.cross(r, force)
    return M

def normalize(vectors):
    """
    Normalize vectors in-place to unit length.

    Modifies the input array so each row vector has magnitude 1.

    Args:
        vectors (np.ndarray): Array of vectors with shape (n, 3).
    """
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]


def get_triangle_normal(triangle):
    """
    Calculate the normal of a single triangle.

    Args:
        triangle (np.ndarray): Coordinates of triangle vertices with shape (3, 3).

    Returns:
        np.ndarray: Unit normal vector of the triangle.
    """
    normal = np.cross(triangle[1] - triangle[0],  triangle[2] - triangle[0])
    return normal / np.linalg.norm(normal)


def get_triangle_normals(triangles):
    """
    Calculate normals for multiple triangles.

    Args:
        triangles (np.ndarray): Array of triangles with shape (n, 3, 3).

    Returns:
        np.ndarray: Array of unit normals with shape (n, 3).
    """
    normals = np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0], axis=1)
    normalize(normals)
    return normals


def get_center_positions(triangles):
    """
    Compute centroid positions of triangles.

    Args:
        triangles (np.ndarray): Array of triangles with shape (n, 3, 3).

    Returns:
        np.ndarray: Array of centroid coordinates with shape (n, 3).
    """
    return np.sum(triangles, axis=1) / 3


def set_normals_pointing_outward(normals, centers):
    """
    Ensure triangle normals point outward from the origin.

    Flips normals whose dot product with their center vector is negative.

    Args:
        normals (np.ndarray): Array of normal vectors with shape (n, 3).
        centers (np.ndarray): Array of triangle centroids with shape (n, 3).

    Returns:
        None
    """
    dot_products = np.sum(normals * centers, axis=1)
    condition = dot_products < 0
    normals[condition] *= (-1)