import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from primitives.primitive import Primitive
from primitives.utility import Utility


class CylinderPrimitive(Primitive):
    def __init__(self, xml_geom):
        super().__init__(xml_geom)
        self._radius = 1.0
        self._height = 1.0
        self._set_attributes_from_xml(xml_geom)

    def _set_attributes_from_xml(self, xml_geom):
        attributes_str = xml_geom.get('size')
        if attributes_str == None:
            return
        self._radius, self._height = Utility.convert_to_float(attributes_str)

    def get_attributes(self):
        return self._radius, self._height
