#*******************************************************************************
# CMakeLists.txt file for the Ark library's math unit tests executable.
#
# Copyright © 2023 Noah Stein. All Rights Reserved.
#*******************************************************************************

include(FetchContent)

FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.13.0
	)

# Generate the file name of a SIMD test suite.
macro(set_simd_test var class)
	set(${var} $<$<NOT:$<STREQUAL:${HAL_SIMD},None>>:${class}${HAL_SIMD}UnitTests.cpp>)
endmacro()

set(gest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

# Determine SIMD platform-specific test files names
set_simd_test(QUAT_SIMD_TESTS Quat)
set_simd_test(VEC2_SIMD_TESTS Vec2)
set_simd_test(VEC4_SIMD_TESTS Vec4)

add_executable(MathTests

	MatrixUnitTests.cpp
	MtxUnitTests.cpp
	Mtx4x4UnitTests.cpp

	VectorUnitTests.cpp
	Vec2UnitTests.cpp
	${VEC2_SIMD_TESTS}
	Vec3UnitTests.cpp
	Vec4UnitTests.cpp
	${VEC4_SIMD_TESTS}

	QuatUnitTests.h
	QuaternionUnitTests.cpp
	QuatUnitTests.cpp
	${QUAT_SIMD_TESTS}
)

	target_link_libraries(MathTests
		PRIVATE
			GTest::GTest
	)

	include(GoogleTest)
	gtest_discover_tests(MathTests DISCOVERY_MODE PRE_TEST)
