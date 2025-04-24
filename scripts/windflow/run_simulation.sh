OUTPUT_DIR=./output

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

if [ -z "$1" ]; then
  echo "Error: No XML path provided."
  echo "Usage: source parser.sh /path/to/static_objects.xml"
  return 1
fi

source run_parser.sh "$1"

if [ -d "$OUTPUT_DIR" ]; then
    source run_openfoam.sh
else
    echo "Directory \$OUTPUT_DIR does not exist, failed to launch simulation."
fi

