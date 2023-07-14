include(FetchContent)

FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.13.0
	)

set(gest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)


add_executable(MathTests 
	VectorUnitTests.cpp
	VecUnitTests.cpp
	Vec2UnitTests.cpp
	Vec4UnitTests.cpp

	QuaternionUnitTests.cpp
	QuatUnitTests.cpp
	)

	target_link_libraries(MathTests
		PRIVATE
			GTest::GTest
	)

	include(GoogleTest)
	gtest_discover_tests(MathTests)
	