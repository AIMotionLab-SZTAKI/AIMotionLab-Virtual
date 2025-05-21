#!/bin/bash

TESTS_FOLDER=./parser/testing

run_python_scripts() {
    dir_path="$1"

    for py_file in "$dir_path"/*.py; do
        if [ -f "$py_file" ]; then
            echo "Running: $py_file"
            python3.10 "$py_file"
        fi
    done
}

for dir in "$TESTS_FOLDER"/*/; do
    if [ -d "$dir" ]; then
      run_python_scripts "$dir"
    fi
done