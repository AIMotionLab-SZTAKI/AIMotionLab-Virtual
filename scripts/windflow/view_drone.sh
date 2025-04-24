#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

if [ -z "$1" ]; then
  echo "Error: No XML path provided."
  echo "Usage: source view_objects.sh /path/to/static_objects.xml /path/to/data.csv"
  return 1
fi


if [ -z "$2" ]; then
  echo "Error: No CSV path provided."
  echo "Usage: source view_objects.sh /path/to/static_objects.xml /path/to/data.csv"
  return 1
fi

DRONE_SCRIPT=../examples/10_windflow.py

pushd ../examples > /dev/null
python3 $DRONE_SCRIPT --xml_path "$1" --csv_path "$2"
popd > /dev/null