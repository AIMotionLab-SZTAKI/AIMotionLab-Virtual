import trimesh
import numpy as np
from itertools import product

from parser_blocks_interface.iinternal_point_calculator import IInternalPointCalculator
from primitives.utility import Utility
from parser_blocks_implementation.space_cuboid import SPACE_CUBOID


class InternalPointCalculator(IInternalPointCalculator):
    def __init__(self):
        self._MAX_DEPTH = 3
        self._DIMENSIONS = np.abs(SPACE_CUBOID.min_point - SPACE_CUBOID.max_point)
        self._meshes = None

    def _get_subdivision_points(self, subdivision_depth):
        numbers = [(i / subdivision_depth) for i in range(1, subdivision_depth, 2)]
        points = np.array(list(product(numbers, numbers, numbers)))
        points = points * self._DIMENSIONS
        points = points + SPACE_CUBOID.min_point
        return points

    def _get_internal_points(self, points):
        internal_point_mask = self._meshes[0].nearest.signed_distance(points) < 0

        for mesh in self._meshes[1:]:
            if np.all(internal_point_mask == False):
                return []

            current_mask = mesh.nearest.signed_distance(points) < 0
            internal_point_mask &= current_mask

        return points[internal_point_mask]

    def get_internal_point(self, primitives):
        self._meshes = [Utility.make_trimesh_object(primitive) for primitive in primitives]

        for i in range(1, self._MAX_DEPTH + 1):
            subdivision_depth = 2 ** i
            subdivision_points = self._get_subdivision_points(subdivision_depth)

            internal_points = self._get_internal_points(subdivision_points)
            if len(internal_points) != 0:
                return internal_points[0]

        raise ValueError('Error: internal point not found, try increasing subdivision density.')
