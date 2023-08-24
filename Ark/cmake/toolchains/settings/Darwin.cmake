# *******************************************************************************
# CMake file for processing settings for macOS builds. This file is to be
# included when building on a Mac without specifying a specific toolchain file
# to use. Most of the settings are handled in included files. This file is
# responsible for evaluating host information and then relying upon setup files
# common with toolchain builds.
#
# Copyright © 2023 Noah Stein. All Rights Reserved.
# *******************************************************************************

# Default macOS to arm64
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

# Include configuration based on CPU processor
include("cmake/toolchains/settings/macOS/macOS-${CMAKE_OSX_ARCHITECTURES}.cmake")
