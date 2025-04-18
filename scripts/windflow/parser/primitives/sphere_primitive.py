import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from primitives.primitive import Primitive
from primitives.utility import Utility


class SpherePrimitive(Primitive):
    def __init__(self, xml_geom):
        super().__init__(xml_geom)
        self._radius = 1.0
        self._set_radius_from_xml(xml_geom)

    def _set_radius_from_xml(self, xml_geom):
        radius_attribute = xml_geom.get('size')
        if radius_attribute == None:
            return
        self._radius = float(radius_attribute)

    def get_radius(self):
        return self._radius
