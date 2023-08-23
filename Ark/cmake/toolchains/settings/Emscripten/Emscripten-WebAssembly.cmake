# *******************************************************************************
# Setup code for Emscripten WebAssembly builds.
#
# Copyright � 2023 Noah Stein. All Rights Reserved.
# *******************************************************************************

set(HAL_SIMD None)

add_compile_definitions(
    gtest_disable_pthreads=ON
)

