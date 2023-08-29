/*************************************************************************
 * @file
 * @brief SSE4-optimized 4-D Vec Unit Tests
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
#define VEC4_TEST_NAMESPACE vec4_sse4_unit_tests

/// @brief @includedoc Test/ParameterizedSuitClass.txt
#define VEC4_TEST_CLASS Vec4Sse4UnitTest


namespace ark::math::test::VEC4_TEST_NAMESPACE
{
	/*********************************************************************
	* @brief @includedoc Test/ParameterizedSuitTypesBrief.txt
	* @details @includedoc Test/ParameterizedSuitTypesDetails.txt
	*/
	using TestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Sse4>,
		Cfg<double, ::ark::hal::simd::Sse4>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "Vec4UnitTests.h"
