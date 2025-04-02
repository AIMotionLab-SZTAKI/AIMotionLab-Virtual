#!/bin/bash

SOURCE_XML=../../aiml_virtual/resources/xml_models/static_objects.xml
OUTPUT_PATH=./output

python ./parser/parser.py \
    $SOURCE_XML \
    $OUTPUT_PATH \
    --merge