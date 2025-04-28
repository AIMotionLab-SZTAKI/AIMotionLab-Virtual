#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

INPUT_XML="$1"

if [ ! -d "$INPUT_XML" ]; then
  INPUT_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
fi

OUTPUT_PATH=./output

python3 ./parser/parser.py \
    "$INPUT_XML" \
    $OUTPUT_PATH \
    --semi-merge