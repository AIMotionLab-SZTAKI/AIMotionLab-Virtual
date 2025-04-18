import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from primitives.primitive import Primitive
from primitives.utility import Utility


class EllipsoidPrimitive(Primitive):
    def __init__(self, xml_geom):
        super().__init__(xml_geom)
        self._radii = np.array([1.0, 1.0, 1.0])
        self._set_radii_from_xml(xml_geom)

    def _set_radii_from_xml(self, xml_geom):
        radii_str = xml_geom.get('size')
        if radii_str == None:
            return
        self._radii = Utility.convert_to_float(radii_str)

    def get_radii(self):
        return self._radii
