import sys
import argparse
import shutil
from pathlib import Path

from parser_blocks_implementation.preprocessor import Preprocessor
from parser_blocks_implementation.xml_object_parser import XMLObjectParser
from parser_blocks_implementation.stl_processor_semi_merge import STLProcessorSemiMerge
from parser_blocks_implementation.stl_processor_separate import STLProcessorSeparate
from parser_blocks_implementation.stl_processor_merge import STLProcessorMerge
from parser_blocks_implementation.bound_checker import BoundChecker
from parser_blocks_implementation.internal_point_calculator import InternalPointCalculator


class Parser:
    def __init__(self, preprocessor, object_parser, stl_processor, bound_checker, point_calculator):
        self._TEMPORARY_NAME = '__temp__.xml'
        self._STL_FILENAME = 'merged'
        self._preprocessor = preprocessor
        self._object_parser = object_parser
        self._stl_processor = stl_processor
        self._bound_checker = bound_checker
        self._point_calc = point_calculator

    def run(self, source_xml_path, output_dir):
        preprocessed_xml_path = Path.cwd() / self._TEMPORARY_NAME
        if preprocessed_xml_path.exists():
            preprocessed_xml_path.write_text('')
        else:
            preprocessed_xml_path.touch()

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()

        self._preprocessor.preprocess_xml(source_xml_path, preprocessed_xml_path)

        primitives = self._object_parser.get_primitives(preprocessed_xml_path)
        preprocessed_xml_path.unlink()

        self._bound_checker.check_mesh_bounds(primitives)

        stl_names = self._stl_processor.generate_stl(primitives, self._STL_FILENAME)

        output_file = output_dir / Path('info.foamInfo')
        with output_file.open('a') as file:
            internal_point = [f"{x:.5g}" for x in self._point_calc.get_internal_point(primitives)]
            file.write(f'({internal_point[0]} {internal_point[1]} {internal_point[2]})\n')

            for stl_filename in stl_names:
                file.write(stl_filename + '\n')

                moved_destination = Path(output_dir) / Path(stl_filename)
                Path(stl_filename).rename(moved_destination)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('source_xml', type=str)
    arg_parser.add_argument('output_dir', type=str)
    arg_parser.add_argument('--separate', action='store_true')
    arg_parser.add_argument('--merge', action='store_true')
    arg_parser.add_argument('--semi-merge', action='store_true')
    args = arg_parser.parse_args()

    if not args.source_xml:
        print('Error: source_xml_path must be specified')
        sys.exit()

    if not args.output_dir:
        print('Error: output_dir_path must be specified')
        sys.exit()

    if sum([args.separate, args.merge, args.semi_merge]) != 1:
        print('Error: stl processor option must be specified: --separate, --merge, --semi-merge')
        sys.exit()

    preprocessor = Preprocessor()
    xml_parser = XMLObjectParser()

    if args.separate:
        stl_processor = STLProcessorSeparate()
    elif args.merge:
        stl_processor = STLProcessorMerge()
    else:
        stl_processor = STLProcessorSemiMerge()

    bound_checker = BoundChecker()
    point_calculator = InternalPointCalculator()

    parser = Parser(
        preprocessor,
        xml_parser,
        stl_processor,
        bound_checker,
        point_calculator)

    parser.run(Path(args.source_xml), Path(args.output_dir))
