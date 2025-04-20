OUTPUT_DIR=./output

source run_parser.sh

if [ -d "$OUTPUT_DIR" ]; then
    source run_openfoam.sh
else
    echo "Directory \$OUTPUT_DIR does not exist, failed to launch simulation."
fi

