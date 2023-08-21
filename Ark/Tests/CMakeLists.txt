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

set_simd_test(QUAT_SIMD_TESTS Quat)
add_executable(MathTests

	VectorUnitTests.cpp
	VecUnitTests.cpp
	Vec2UnitTests.cpp
	Vec4UnitTests.cpp

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
