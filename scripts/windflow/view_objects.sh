#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

INPUT_XML="$1"

if [ -z "$INPUT_XML" ]; then
  INPUT_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
fi

VIEW_SCRIPT=11_test_view_static_objects.py

pushd ../examples
python3 $VIEW_SCRIPT --xml_path "$INPUT_XML"
popd