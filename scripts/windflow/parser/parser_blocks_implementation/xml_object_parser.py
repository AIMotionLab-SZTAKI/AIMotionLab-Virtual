import xml.etree.ElementTree as ET
import numpy as np

from parser_blocks_interface.ixml_object_parser import IXMLObjectParser
from primitives.utility import Utility
from primitives.primitive_factory import PrimitiveFactory


class XMLObjectParser(IXMLObjectParser):
    def __init__(self):
        self._primitives = []
        self._asset_map = {}
        self._factory = None

    def get_primitives(self, source_xml_path):
        self._populate_asset_map(source_xml_path)
        self._factory = PrimitiveFactory(self._asset_map)

        tree = ET.parse(source_xml_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        self._process_nodes(worldbody, np.eye(4))

        return self._primitives

    def _populate_asset_map(self, source_xml_path):
        tree = ET.parse(source_xml_path)
        root = tree.getroot()
        assets = root.find('asset')

        if assets is not None:
            for asset in assets:
                mesh_name = asset.get('name')
                self._asset_map[mesh_name] = asset

    def _process_nodes(self, node, parent_transform):
        if node.tag == 'geom':
            primitive = self._factory.make_primitive(node)
            primitive.apply_parent_transform(parent_transform)
            self._primitives.append(primitive)
            return

        body_transform = Utility.get_parent_transform_from_body(node)
        for child in node:
            self._process_nodes(child, parent_transform @ body_transform)
