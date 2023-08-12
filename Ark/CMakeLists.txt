cmake_minimum_required(VERSION 3.26.0)
project(Ark VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Temporary until toolchains arrive
set(HAL_SIMD Avx2)
add_compile_definitions(
	HAL_SIMD=${HAL_SIMD}
)
add_compile_options(
	/bigobj
)


include_directories(..)

enable_testing()

add_subdirectory(Tests)
