import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from pathlib import Path


from primitives.primitive import Primitive
from primitives.utility import Utility


class MeshPrimitive(Primitive):
    def __init__(self, xml_geom, source_path, asset):
        super().__init__(xml_geom)
        self._scale    = np.array([1.0, 1.0, 1.0])
        self._xml_path = None
        self._set_attributes_path(source_path, asset)
        
        scale_matrix = trimesh.transformations.scale_matrix(self._scale[0], direction=[1, 0, 0]) @ \
                       trimesh.transformations.scale_matrix(self._scale[1], direction=[0, 1, 0]) @ \
                       trimesh.transformations.scale_matrix(self._scale[2], direction=[0, 0, 1])
        self._transform = self._transform @ scale_matrix


    def _set_attributes_path(self, source_path, asset):
        stl_file_path = Path(asset.get('file'))
        if not stl_file_path.is_absolute():
            stl_file_path = source_path / stl_file_path
        self._xml_path = stl_file_path

        scale_attribute = asset.get('scale') 
        if scale_attribute != None:
            self._scale = Utility.convert_to_float(scale_attribute)


    def get_stl_path(self):
        return sefl._xml_path
