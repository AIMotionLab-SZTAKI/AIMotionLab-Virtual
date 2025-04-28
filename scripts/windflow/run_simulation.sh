OUTPUT_DIR=./output

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

INPUT_XML="$1"

if [ ! -d "$INPUT_XML" ]; then
  INPUT_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
fi

source run_parser.sh "$INPUT_XML"

if [ -d "$OUTPUT_DIR" ]; then
    source run_openfoam.sh
else
    echo "Directory \$OUTPUT_DIR does not exist, failed to launch simulation."
fi

