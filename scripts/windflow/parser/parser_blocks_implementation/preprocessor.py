import sys
import re

from parser_blocks_interface.ipreprocessor import IPreprocessor

class Preprocessor(IPreprocessor):
    def __init__(self):
        self._SOURCE_DIR = None


    def preprocess_xml(self, source_xml_path, final_xml_path):
        if not final_xml_path.exists():
            print('Error: output file path does not exist')
            sys.exit()

        self._SOURCE_DIR = source_xml_path.parent

        output_file = open(final_xml_path, 'a')
        self._opening_new_file(source_xml_path, output_file)
        output_file.close()


    def _opening_new_file(self, filename, output_file):
        with open(filename, 'r') as file:
            for line in file:
                if 'include' in line:
                    extracted_path = self._get_path_from_line(line)
                    self._opening_new_file(self._SOURCE_DIR / extracted_path, output_file)
                else:
                    output_file.write(line)


    def _get_path_from_line(self, line):
        splitted_line = re.split(r'["\']', line)
        for word in splitted_line:
            if '.xml' in word:
                return word
        raise ValueError('Error: file constains keyword include but does not name a .xml file')
