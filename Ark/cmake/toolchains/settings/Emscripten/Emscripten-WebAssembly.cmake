# *******************************************************************************
# Setup code for Emscripten WebAssembly builds.
#
# Copyright � 2023 Noah Stein. All Rights Reserved.
# *******************************************************************************

set(HAL_SIMD Wasm128)

add_compile_definitions(
    gtest_disable_pthreads=ON
)

add_compile_options(
    -msimd128   # Compile with 128-bit SIMD extension
)
