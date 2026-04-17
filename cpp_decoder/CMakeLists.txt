cmake_minimum_required(VERSION 3.15)
project(DecoderPythonInterface VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# 1. Tell pybind11 to use the new FindPython instead of the old FindPythonLibs
set(PYBIND11_FINDPYTHON ON)

# 2. Explicitly find Python (Development component is required for headers)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 3. Now find pybind11
find_package(pybind11 REQUIRED)
# --- THE FIX ENDS HERE ---

find_package(fmt CONFIG REQUIRED)

pybind11_add_module(cpp_tbcc_decoder 
    src/bindings.cpp 
    src/minHeap.cpp
    src/tbccDecoder.cpp
    src/feedForwardTrellis.cpp # Don't forget to include this!
)

target_include_directories(cpp_tbcc_decoder PRIVATE include)

target_link_libraries(cpp_tbcc_decoder PRIVATE 
    pybind11::module 
    fmt::fmt
)