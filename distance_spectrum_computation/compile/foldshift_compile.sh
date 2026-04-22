#!/bin/bash

INPUT_FILE="distance_spectrum.cu"
OUTPUT_FILE="lib/libfoldshift.so"

echo "Compiling $INPUT_FILE..."

# Create output directory if it doesn't exist
mkdir -p ../lib

nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_75,code=sm_75 distance_spectrum.cu -o ../lib/libfoldshift.so

if [ $? -eq 0 ]; then
    echo "Compilation successful: $OUTPUT_FILE"
else
    echo "Compilation failed."
    exit 1
fi