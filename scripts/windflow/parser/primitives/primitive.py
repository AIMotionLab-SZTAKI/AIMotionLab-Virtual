import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


from primitives.utility import Utility
from primitives.primitive_description import PrimitiveDescription


class Primitive:
    def __init__(self, xml_geom):
        self._name      = xml_geom.get('name')
        self._type      = xml_geom.get('type')
        self._rgba      = Utility.extract_rgba_from_xml(xml_geom)
        self._position  = Utility.extract_position_from_xml(xml_geom)
        self._rotation  = Utility.extract_rotation_from_xml(xml_geom)
        self._transform = Utility.get_translation_transform_matrix(self._position) @ \
                          Utility.get_orientation_transform_matrix(self._rotation)


    def get_description(self):
        return PrimitiveDescription(
            self._name,
            self._type,
            self._rgba,
            self._position,
            self._rotation
        )


    def apply_parent_transform(self, parent_transform_matrix):
        self._transform = parent_transform_matrix @ self._transform


    def get_transform(self):
        return self._transform
