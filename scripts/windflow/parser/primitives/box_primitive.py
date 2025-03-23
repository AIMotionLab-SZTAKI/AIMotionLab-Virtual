import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from primitives.primitive import Primitive
from primitives.utility import Utility

class BoxPrimitive(Primitive):
    def __init__(self, xml_geom):
        super().__init__(xml_geom)
        self._extents = np.array([1.0, 1.0, 1.0])
        self._set_extents_from_xml(xml_geom)

    
    def _set_extents_from_xml(self, xml_geom):
        extents_attribute = xml_geom.get('size')
        if extents_attribute == None:
            return
        self._extents = Utility.convert_to_float(extents_attribute)


    def get_extents(self):
        return self._extents
