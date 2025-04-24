#!/bin/bash

if [ "$(basename "$PWD")" != "windflow" ]; then
  echo "Error: You must run this script from the 'windflow' directory."
  return 1
fi

OUTPUT_DIR=./output
FOAM_DICT_NAME="foam_files"

FOAM_PROJ_FILE=./$FOAM_DICT_NAME/open.foam
OUTPUT_CSV_PATH=./data.csv

clone_archive() {
    local archive=.foam_archive
    if [ -d "$FOAM_DICT_NAME" ]; then
        rm -rf "$FOAM_DICT_NAME"
    fi
    cp -r "$archive" "$FOAM_DICT_NAME"
}

populate_snappy_config_dicts() {
    local snappy_file="$FOAM_DICT_NAME"/mesh/system/snappyHexMeshDict
    local features_file="$FOAM_DICT_NAME"/mesh/system/surfaceFeaturesDict

    local snappy_features="( "
    local internal_point=$(head "$OUTPUT_DIR"/info.foamInfo -n 1)

    for file in "$OUTPUT_DIR"/*.stl; do
        filename_stl=$(basename "$file")
        filename=${filename_stl%.stl}

        snappy_features+="{ file \"$filename.eMesh\"; level 0; } "	

        foamDictionary "$snappy_file" -entry geometry/"$filename" -add "{}"
        foamDictionary "$snappy_file" -entry geometry/"$filename"/type -add "triSurfaceMesh"
        foamDictionary "$snappy_file" -entry geometry/"$filename"/file -add "\"$filename_stl\""

        foamDictionary "$features_file" -entry "${filename}" -add "{}"
        foamDictionary "$features_file" -entry "${filename}"/surfaces -add "( \"$filename_stl\" )"
        foamDictionary "$features_file" -entry "${filename}"/includedAngle -add 150

        foamDictionary "$snappy_file" -entry castellatedMeshControls/refinementSurfaces/"$filename" -add "{}"
        foamDictionary "$snappy_file" -entry castellatedMeshControls/refinementSurfaces/"$filename"/level -set "(0 0)"
        
        local foam_stl_dir="$FOAM_DICT_NAME"/mesh/constant/triSurface/"$filename_stl"
        cp "$file" "$foam_stl_dir"
    done

    snappy_features+=" )"
    foamDictionary "$snappy_file" -entry castellatedMeshControls/features -add "$snappy_features"
    foamDictionary "$snappy_file" -entry castellatedMeshControls/locationInMesh -add "$internal_point"
}

populate_patch_field_dicts() {
    local p_file="$FOAM_DICT_NAME"/"0"/p
    local U_file="$FOAM_DICT_NAME"/"0"/U
    local nut_file="$FOAM_DICT_NAME"/"0"/nut
    local nuTilda_file="$FOAM_DICT_NAME"/"0"/nuTilda


    for file in "$OUTPUT_DIR"/*.stl; do
        filename_stl=$(basename "$file")
        filename=${filename_stl%.stl}

        foamDictionary "$p_file" -entry boundaryField/"$filename" -add "{}"
        foamDictionary "$p_file" -entry boundaryField/"$filename"/type -add "zeroGradient"

        foamDictionary "$U_file" -entry boundaryField/"$filename" -add "{}"
        foamDictionary "$U_file" -entry boundaryField/"$filename"/type -add "noSlip"

        foamDictionary "$nut_file" -entry boundaryField/"$filename" -add "{}"
        foamDictionary "$nut_file" -entry boundaryField/"$filename"/type -add "nutUSpaldingWallFunction"
        foamDictionary "$nut_file" -entry boundaryField/"$filename"/value -add "uniform 0"

        foamDictionary "$nuTilda_file" -entry boundaryField/"$filename" -add "{}"
        foamDictionary "$nuTilda_file" -entry boundaryField/"$filename"/type -add "fixedValue"
        foamDictionary "$nuTilda_file" -entry boundaryField/"$filename"/value -add "uniform 0"
    done
}

run_simulation() {
    if [ -d "$OUTPUT_DIR" ]; then
        rm -rf "$OUTPUT_DIR"
    fi

    pushd "$FOAM_DICT_NAME"/mesh > /dev/null
    source runMesh.sh
    popd

    pushd "$FOAM_DICT_NAME" > /dev/null
    source runSim.sh
    popd
}

postprocess() {
  /opt/paraview/bin/pvpython paraview_export.py --foam_proj "$FOAM_PROJ_FILE" --output_csv "$OUTPUT_CSV_PATH"
}

clone_archive
populate_snappy_config_dicts
populate_patch_field_dicts
run_simulation
postprocess