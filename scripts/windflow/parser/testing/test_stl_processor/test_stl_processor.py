import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from parser_blocks_implementation import stl_processor_merge, stl_processor_semi_merge, stl_processor_separate, \
    xml_object_parser
from pathlib import Path


class TestSTLProcessors(unittest.TestCase):
    STL_FILES_DIR = Path('./stl_files')

    def setUp(self):
        self.STL_FILENAME = './stl_files/test'
        self.NUMBER_OF_SEPARATE_FILES = 6
        self.NUMBER_OF_MERGED_FILES = 4

        self.input_file = Path('./example.xml')
        self.parser = xml_object_parser.XMLObjectParser()
        self.primitives = self.parser.get_primitives(self.input_file)

        self.merge = stl_processor_merge.STLProcessorMerge()
        self.semi_merge = stl_processor_semi_merge.STLProcessorSemiMerge()
        self.separate = stl_processor_separate.STLProcessorSeparate()

    @classmethod
    def clear_dir(cls):
        for file in TestSTLProcessors.STL_FILES_DIR.iterdir():
            if file.is_file():
                file.unlink()

    def test_merge_valid_number_of_files(self):
        TestSTLProcessors.clear_dir()
        self.merge.generate_stl(self.primitives, self.STL_FILENAME)
        path = Path(self.STL_FILENAME + '.stl')
        self.assertTrue(path.exists())

    def test_separate_valid_number_of_files(self):
        TestSTLProcessors.clear_dir()
        self.separate.generate_stl(self.primitives, self.STL_FILENAME)
        file_count = len([f for f in os.listdir(TestSTLProcessors.STL_FILES_DIR) if
                          os.path.isfile(os.path.join(TestSTLProcessors.STL_FILES_DIR, f))])
        self.assertEqual(file_count, self.NUMBER_OF_SEPARATE_FILES)

    def test_semi_merge_valid_number_of_files(self):
        TestSTLProcessors.clear_dir()
        self.semi_merge.generate_stl(self.primitives, self.STL_FILENAME)
        file_count = len([f for f in os.listdir(TestSTLProcessors.STL_FILES_DIR) if
                          os.path.isfile(os.path.join(TestSTLProcessors.STL_FILES_DIR, f))])
        self.assertEqual(file_count, self.NUMBER_OF_MERGED_FILES)

    @classmethod
    def tearDownClass(cls):
        TestSTLProcessors.clear_dir()
