#!/bin/bash

rm -rf constant/polyMesh
cp -r mesh/constant/polyMesh constant/

cp utils/momentumTransport_init constant/momentumTransport
cp utils/controlDict_init system/controlDict

simpleFoam
