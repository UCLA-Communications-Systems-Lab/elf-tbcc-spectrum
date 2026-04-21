cmake_minimum_required(VERSION 3.15)
project(DecoderPythonInterface VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Auto-detect pybind11 from pip-installed location if not already found
if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_DIR_RESULT
    )
    if(NOT pybind11_DIR_RESULT EQUAL 0)
        message(WARNING "Could not detect pybind11 via pip. Falling back to system search.")
    else()
        # Normalize backslashes to forward slashes for CMake
        file(TO_CMAKE_PATH "${pybind11_DIR}" pybind11_DIR)
        message(STATUS "Found pybind11 cmake dir via pip: ${pybind11_DIR}")
    endif()
endif()

find_package(pybind11 REQUIRED)

find_package(fmt CONFIG REQUIRED)

pybind11_add_module(cpp_tbcc_decoder 
    src/bindings.cpp 
    src/minHeap.cpp
    src/tbccDecoder.cpp
    src/feedForwardTrellis.cpp
)

target_include_directories(cpp_tbcc_decoder PRIVATE include)

target_link_libraries(cpp_tbcc_decoder PRIVATE 
    pybind11::module 
    fmt::fmt
)