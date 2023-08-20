#*******************************************************************************
# Setup code common to host and toolchain builds for 64-bit x86 Windows builds.
#
# Copyright © 2023 Noah Stein. All Rights Reserved.
#*******************************************************************************

set(HAL_SIMD Avx2)

add_compile_options(
	/bigobj
)

