import numpy as np
import trimesh

class SpaceCuboid:
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point


    def is_mesh_inside_space(self, mesh):
        cube_min_point, cube_max_point = mesh.bounds 
        return np.all(self.min_point < cube_min_point) and np.all(cube_max_point < self.max_point)

SPACE_MIN_X = -4.17
SPACE_MAX_X = 1.27

SPACE_MIN_Y = -3.06
SPACE_MAX_Y = 3.07

SPACE_MIN_Z = 0
SPACE_MAX_Z = 2.7

SPACE_CUBOID = SpaceCuboid(
    np.array([SPACE_MIN_X, SPACE_MIN_Y, SPACE_MIN_Z]),
    np.array([SPACE_MAX_X, SPACE_MAX_Y, SPACE_MAX_Z])
)
