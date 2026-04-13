#!/bin/bash

INPUT_FILE="combiner.cu"
OUTPUT_FILE="lib/libtrellis.so"

echo "Compiling $INPUT_FILE..."

# Create output directory if it doesn't exist
mkdir -p lib

# for RTX 40-series
# nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_89,code=sm_89 combiner.cu -o lib/libtrellis.so

# for T4 GPU
# nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_75,code=sm_75 combiner.cu -o lib/libtrellis.so
nvcc -shared -Xcompiler -fPIC -O3 \
    -gencode arch=compute_89,code=sm_89 \
    "$INPUT_FILE" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Compilation successful: $OUTPUT_FILE"
else
    echo "Compilation failed."
    exit 1
fi