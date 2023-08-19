/*************************************************************************
 * @file
 * @brief Quat Neon32-optimized Implementation Unit Tests
 *
 * @details Unit tests for the None32-optimized Quat class implementations.
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
#define QUAT_TEST_NAMESPACE quat_neon32_unit_tests
#define QUAT_TEST_CLASS QuatNeon32UnitTest

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	using QuatTestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Neon32>,
		Cfg<double, ::ark::hal::simd::Neon32>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "QuatUnitTests.h"
