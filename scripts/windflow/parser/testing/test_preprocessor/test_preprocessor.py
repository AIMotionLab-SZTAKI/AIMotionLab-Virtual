import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from parser_blocks_implementation import preprocessor
from pathlib import Path


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.input_file = Path('./example_1.xml')
        self.included_file1 = Path('./example_2.xml')
        self.included_file2 = Path('./example_3.xml')
        self.output_file = Path('./__temp__.xml')
        self.output_file.touch()

    def test_valid_preprocessing(self):
        preproc = preprocessor.Preprocessor()
        preproc.preprocess_xml(self.input_file, self.output_file)

        output_lines = []
        with self.output_file.open('r') as file:
            for line in file:
                if 'include' in line:
                    self.output_file.unlink()
                    self.fail()
                output_lines.append(line.strip())

        self.check_file_included(self.input_file, output_lines)
        self.check_file_included(self.included_file1, output_lines)

        # if the previous tasks did not fail, then all tests are accepted
        self.output_file.unlink()
        self.assertTrue(True)

    def check_file_included(self, filepath, output_lines):
        with filepath.open('r') as file:
            for line in file:
                line = line.strip()
                if 'include' in line:
                    continue
                if not (line in output_lines):
                    print(f'[{filepath.name}]: line not in output:', line)
                    print(output_lines)
                    self.output_file.unlink()
                    self.fail()

if __name__ == '__main__':
    unittest.main()