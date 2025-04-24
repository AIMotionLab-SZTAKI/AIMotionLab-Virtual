#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

if [ -z "$1" ]; then
  echo "Error: No XML path provided."
  echo "Usage: source view_objects.sh /path/to/static_objects.xml"
  return 1
fi

VIEW_SCRIPT=../examples/11_test_view_static_objects.py

pushd ../examples > /dev/null
python3 $VIEW_SCRIPT --xml_path "$1"
popd > /dev/null