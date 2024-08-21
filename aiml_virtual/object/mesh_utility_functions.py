import numpy as np

def normalize(vectors):
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]


def get_triangle_normal(triangle):
    normal = np.cross(triangle[1] - triangle[0],  triangle[2] - triangle[0])
    return normal / np.linalg.norm(normal)


def get_triangle_normals(triangles):
    normals = np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0], axis=1)
    normalize(normals)
    return normals


def get_mid_point(v1, v2):
    return (v1 + v2) / 2


def get_center_positions(triangles):
    return np.sum(triangles, axis=1) / 3


def set_normals_pointing_outward(normals, centers):
    dot_products = np.sum(normals * centers, axis=1)
    condition = dot_products < 0
    normals[condition] *= (-1)