#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

if [ -z "$1" ]; then
  echo "Error: No AIMotionLab-Virtual mount point provided."
  echo "Usage: source start_container.sh /ABSOLUTE/PATH/to/dir"
  return 1
fi

xhost +local:docker

docker run -it \
  --rm \
  --privileged \
  --hostname "windflow" \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$1":/home/app \
  szabokrisztian/thesis_environment:latest

xhost -local:docker