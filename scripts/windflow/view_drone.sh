#!/bin/bash

DRONE_SCRIPT=../examples/10_windflow.py

pushd ../examples
python3 $DRONE_SCRIPT
popd