import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from parser_blocks_implementation import xml_object_parser
from pathlib import Path


class TestXMLObjectParser(unittest.TestCase):
    def get_object_dict(self):
        dict = {}
        for primitive in self.primitives:
            primitive_type = primitive.get_description().type
            dict[primitive_type] = primitive
        return dict

    def setUp(self):
        self.NUMBER_OF_PRIMITIVES = 6
        self.PRIMITIVE_TYPES = [
            'box', 'sphere', 'cylinder', 'capsule', 'ellipsoid', 'mesh'
        ]
        self.OBJECT_DIMENSIONS = {
            'box': [0.5, 0.5, 0.5],
            'sphere': 0.5,
            'cylinder': [0.5, 0.4],
            'capsule': [0.5, 0.3],
            'ellipsoid': [0.5, 0.3, 0.4]
        }

        self.input_file = Path(__file__).parent / Path('./example.xml')
        self.parser = xml_object_parser.XMLObjectParser()
        self.primitives = self.parser.get_primitives(self.input_file)
        self.primitives_dict = self.get_object_dict()

    def test_valid_number_of_objects(self):
        self.assertEqual(len(self.primitives), self.NUMBER_OF_PRIMITIVES)

    def test_valid_object_types(self):
        all_object_types_valid = True
        for primitive in self.primitives:
            if not (primitive.get_description().type in self.PRIMITIVE_TYPES):
                all_object_types_valid = False
                break
        self.assertTrue(all_object_types_valid)

    def test_valid_object_dimensions_box(self):
        box_dimensions = self.primitives_dict['box'].get_extents()
        self.assertEqual(box_dimensions, self.OBJECT_DIMENSIONS['box'])

    def test_valid_object_dimensions_sphere(self):
        sphere_radius = self.primitives_dict['sphere'].get_radius()
        self.assertEqual(sphere_radius, self.OBJECT_DIMENSIONS['sphere'])

    def test_valid_object_dimensions_cylinder(self):
        cylinder_radius, cylinder_height = self.primitives_dict['cylinder'].get_attributes()
        self.assertEqual(cylinder_radius, self.OBJECT_DIMENSIONS['cylinder'][0])
        self.assertEqual(cylinder_height, self.OBJECT_DIMENSIONS['cylinder'][1])

    def test_valid_object_dimensions_capsule(self):
        capsule_radius, capsule_height = self.primitives_dict['capsule'].get_attributes()
        self.assertEqual(capsule_radius, self.OBJECT_DIMENSIONS['capsule'][0])
        self.assertEqual(capsule_height, self.OBJECT_DIMENSIONS['capsule'][1])

    def test_valid_object_dimensions_ellipsoid(self):
        ellipsoid_radii = self.primitives_dict['ellipsoid'].get_radii()
        self.assertEqual(ellipsoid_radii, self.OBJECT_DIMENSIONS['ellipsoid'])

if __name__ == '__main__':
    unittest.main()