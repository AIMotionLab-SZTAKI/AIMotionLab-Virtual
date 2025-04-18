from parser_blocks_interface.istl_processor import ISTLProcessor
from primitives.utility import Utility


class STLProcessorSeparate(ISTLProcessor):
    def __init__(self):
        self._meshes = []
        self._stl_names = []

    def generate_stl(self, primitives, stl_filename):
        for i, primitive in enumerate(primitives):
            mesh = Utility.make_trimesh_object(primitive)

            stl_name = stl_filename + '_separate' + str(i) + '.stl'
            self._stl_names.append(stl_name)
            mesh.export(stl_name)

        return self._stl_names
