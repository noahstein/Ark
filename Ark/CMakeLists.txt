#*******************************************************************************
# Main CMakeLists.txt file for the Ark library.
#
# Copyright © 2023 Noah Stein. All Rights Reserved.
#*******************************************************************************

cmake_minimum_required(VERSION 3.26.0)

# If no toolchain file specified, platform-specific configuration for host machine
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	include("cmake/toolchains/settings/${CMAKE_HOST_SYSTEM_NAME}.cmake")
endif()

# Handle Emscripten configuration
if (EMSCRIPTEN)
	include("cmake/toolchains/settings/Emscripten.cmake")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Ark VERSION 0.1)

# Set compiler definitions for standard architectural components
add_compile_definitions(
	HAL_SIMD=${HAL_SIMD}
)

include_directories(..)

enable_testing()

add_subdirectory(Tests)
