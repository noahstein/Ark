#*******************************************************************************
# Setup code common to host and toolchain builds for Arm64 MacOS builds.
#
# Copyright © 2023 Noah Stein. All Rights Reserved.
#*******************************************************************************

set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
set(HAL_SIMD Neon64)
