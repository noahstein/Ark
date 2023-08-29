/*************************************************************************
 * @file
 * @brief SSE2-optimized 2-D Vec Unit Tests
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
#define VEC2_TEST_NAMESPACE vec2_sse2_unit_tests

/// @brief @includedoc Test/ParameterizedSuitClass.txt
#define VEC2_TEST_CLASS Vec2Sse2UnitTest


namespace ark::math::test::VEC2_TEST_NAMESPACE
{
	/*********************************************************************
	* @brief @includedoc Test/ParameterizedSuitTypesBrief.txt
	* @details @includedoc Test/ParameterizedSuitTypesDetails.txt
	*/
	using TestTypes = ::testing::Types
	<
		Cfg<double, ::ark::hal::simd::Sse2>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "Vec2UnitTests.h"
