cmake_minimum_required(VERSION 3.18)
project(main VERSION 0.0.2 DESCRIPTION "Main Executable for Neural Text to Speech" LANGUAGES CXX)

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Determine the appropriate ONNX Runtime directory based on the operating system
if(APPLE)
    set(ONNXRUNTIME_DIR "onnxruntime-osx-arm64-1.20.0/")
elseif(WIN32)
    set(ONNXRUNTIME_DIR "onnxruntime-win-x64-1.20.0/")
else()
    message(FATAL_ERROR "Unsupported operating system for ONNX Runtime selection.")
endif()

# Locate ONNX Runtime library
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)

# Set compiler options to support UTF-8 encoding
if(MSVC)
    add_compile_options("/utf-8")
else()
    add_compile_options("-finput-charset=UTF-8" "-fexec-charset=UTF-8")
endif()

# Include directories for ONNX Runtime and project sources
include_directories(
    include/
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add the main source files
add_executable(main src/main.cpp src/phonemizer.cpp src/cleaners.cpp)

# Link ONNX Runtime and necessary libraries
target_link_libraries(main ${ONNXRUNTIME_LIB})

# For macOS, link additional libraries as needed
if(APPLE)
    target_link_libraries(main "-framework CoreFoundation" "-framework Foundation")
endif()

# Additional linker options for non-Apple platforms
if(NOT APPLE)
    target_link_options(main PRIVATE LINKER:--build-id=none)
    target_compile_options(main PRIVATE "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=main")
endif()

# Set the output directory for the executable
set_target_properties(
    main
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Compile flags - enable all warnings except -Wsign-compare
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main PRIVATE -Wall -Wextra -Wno-sign-compare)
elseif(MSVC)
    target_compile_options(main PRIVATE /W4)
endif()

# Set RPATH to find ONNX Runtime library at runtime
set_target_properties(main PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")

# Output messages for confirmation
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "ONNX Runtime directory: ${ONNXRUNTIME_DIR}")
