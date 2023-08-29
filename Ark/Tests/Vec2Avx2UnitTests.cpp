/*************************************************************************
 * @file
 * @brief AVX2-optimized 2-D Vec Unit Tests
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

/// @brief @includedoc Test/ParameterizedSuitNamespace.txt
#define VEC2_TEST_NAMESPACE vec2_av2x_unit_tests

/// @brief @includedoc Test/ParameterizedSuitClass.txt
#define VEC2_TEST_CLASS Vec2Avx2UnitTest


namespace ark::math::test::VEC2_TEST_NAMESPACE
{
	/*********************************************************************
	* @brief @includedoc Test/ParameterizedSuitTypesBrief.txt
	* @details @includedoc Test/ParameterizedSuitTypesDetails.txt
	*/
	using TestTypes = ::testing::Types
	<
		Cfg<double, ::ark::hal::simd::Avx2>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "Vec2UnitTests.h"
