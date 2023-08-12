/*************************************************************************
* @file
 * @brief Quat AVX-optimized Implementation Unit Tests
 *
 * @details Unit tests for the AVX-optimized Quat class implementations.
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
#define QUAT_TEST_NAMESPACE quat_avx_unit_tests
#define QUAT_TEST_CLASS QuatAvxUnitTest

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	using QuatTestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Avx>,
		Cfg<double, ::ark::hal::simd::Avx>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "QuatUnitTests.h"
