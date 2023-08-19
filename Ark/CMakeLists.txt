cmake_minimum_required(VERSION 3.26.0)

# set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Ark VERSION 0.1)

# Temporary until toolchains arrive
set(HAL_SIMD Neon32)
add_compile_definitions(
	HAL_SIMD=${HAL_SIMD}
)

# macOS
add_compile_options(
	#-mfloat-abi=hard
)

# Windows
#add_compile_options(
#	/bigobj
#)


include_directories(..)

enable_testing()

add_subdirectory(Tests)
