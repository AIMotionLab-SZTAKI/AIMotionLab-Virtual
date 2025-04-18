import trimesh

from parser_blocks_interface.istl_processor import ISTLProcessor
from primitives.utility import Utility


class STLProcessorMerge(ISTLProcessor):
    def __init__(self):
        self._meshes = []

    def generate_stl(self, primitives, stl_filename):
        for primitive in primitives:
            mesh = Utility.make_trimesh_object(primitive)
            self._meshes.append(mesh)

        merged_mesh = trimesh.boolean.union(self._meshes)

        stl_name = stl_filename + '.stl'
        merged_mesh.export(stl_name)
        return [stl_name]
