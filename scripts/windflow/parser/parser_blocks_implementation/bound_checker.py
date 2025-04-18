import sys

from parser_blocks_interface.ibound_checker import IBoundChecker
from primitives.utility import Utility
from parser_blocks_implementation.space_cuboid import SPACE_CUBOID


class BoundChecker(IBoundChecker):
    def check_mesh_bounds(self, primitives):
        for primitive in primitives:
            mesh = Utility.make_trimesh_object(primitive)

            is_mesh_inside_space = SPACE_CUBOID.is_mesh_inside_space(mesh)
            if not is_mesh_inside_space:
                print('Error: mesh is outside space. Mesh info:')
                primitive.get_description().print()
                sys.exit()
