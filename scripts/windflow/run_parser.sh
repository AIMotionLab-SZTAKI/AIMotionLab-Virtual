#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

INPUT_XML="$1"
OUTPUT_PATH=./output
MERGE_MODE="${MERGE_MODE:---semi-merge}"

if [ ! -d "$INPUT_XML" ]; then
  INPUT_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
fi


python3 ./parser/parser.py \
    "$INPUT_XML" \
    $OUTPUT_PATH \
    "$MERGE_MODE"