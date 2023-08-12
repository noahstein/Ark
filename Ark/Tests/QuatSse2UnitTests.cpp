/*************************************************************************
 * @file
 * @brief Quat SSE2-optimized Implementation Unit Tests
 *
 * @details Unit tests for the SSE2-optimized Quat class implementations.
 * As with other Quat implementations, this file contains some
 * configuration information used by QuatUnitTests.h, the file containing
 * the actual tests written in a class-independent fashion relying upon
 * preprocessor parameters.
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Cfg.h"


//************************************************************************
//  Configuration
//************************************************************************
#define QUAT_TEST_NAMESPACE quat_sse2_unit_tests
#define QUAT_TEST_CLASS QuatSse2UnitTest

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	using QuatTestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Sse2>,
		Cfg<double, ::ark::hal::simd::Sse2>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "QuatUnitTests.h"
