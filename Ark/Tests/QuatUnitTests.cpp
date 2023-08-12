/*************************************************************************
 * @file
 * @brief Quat Unoptimized Implementation Unit Tests
 *
 * @details Unit tests for the baseline Quat class implementation. As with 
 * SIMD-optimized Quat classes, this file contains some configuration 
 * information used by QuatUnitTests.h that contain parameterized tests.
 *
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

 //************************************************************************
 //  Dependencies
 //************************************************************************
#include "gtest/gtest.h"
#include "Cfg.h"


//************************************************************************
//  Configuration
//************************************************************************
#define QUAT_TEST_NAMESPACE quat_unit_tests
#define QUAT_TEST_CLASS QuatUnitTest

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	using QuatTestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::None>,
		Cfg<double, ::ark::hal::simd::None>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "QuatUnitTests.h"
