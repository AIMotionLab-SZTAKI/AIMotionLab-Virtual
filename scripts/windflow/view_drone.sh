#!/bin/bash

DRONE_SCRIPT=10_windflow.py

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

if [ -z "$1" ]; then
    echo "Error: No CSV path provided."
    echo "Usage: source view_objects.sh /path/to/data.csv or source view_objects.sh /path/to/objects.xml /path/to/data.csv"
    return 1
fi

INPUT_XML="$1"

if [ -z "$2" ]; then
  INPUT_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
  pushd ../examples
  python3 "$DRONE_SCRIPT" --xml_path "$INPUT_XML" --csv_path "$1"
  popd
else
  pushd ../examples
  python3 "$DRONE_SCRIPT" --xml_path "$1" --csv_path "$2"
  popd
fi