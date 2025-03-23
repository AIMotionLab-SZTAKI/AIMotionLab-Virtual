import xml.etree.ElementTree as ET

from primitives.box_primitive import BoxPrimitive
from primitives.capsule_primitive import CapsulePrimitive
from primitives.cylinder_primitive import CylinderPrimitive
from primitives.ellipsoid_primitive import EllipsoidPrimitive
from primitives.sphere_primitive import SpherePrimitive
from primitives.mesh_primitive import MeshPrimitive

class PrimitiveFactory:
    def __init__(self, mesh_assets, source_path):
        self._mesh_assets = mesh_assets
        self._source_path = source_path


    def make_primitive(self, xml_geom):
        geom_type = xml_geom.get('type')

        match geom_type:
            case 'box':
                return BoxPrimitive(xml_geom)
            case 'sphere':
                return SpherePrimitive(xml_geom)
            case 'capsule':
                return CapsulePrimitive(xml_geom)
            case 'cylinder':
                return CylinderPrimitive(xml_geom)
            case 'ellipsoid':
                return EllipsoidPrimitive(xml_geom)
            case 'mesh':
                mesh_name = xml_geom.get('mesh')
                return MeshPrimitive(xml_geom, self._source_path, self._mesh_assets[mesh_name])
        raise ValueError('Error: illegal geom type')
