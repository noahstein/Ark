/*************************************************************************
 * @file
 * @brief AVX2-optimized 4-D Vec Unit Tests
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
#define VEC4_TEST_NAMESPACE vec4_Avx2_unit_tests

/// @brief @includedoc Test/ParameterizedSuitClass.txt
#define VEC4_TEST_CLASS Vec4Avx2UnitTest


namespace ark::math::test::VEC4_TEST_NAMESPACE
{
	/*********************************************************************
	* @brief @includedoc Test/ParameterizedSuitTypesBrief.txt
	* @details @includedoc Test/ParameterizedSuitTypesDetails.txt
	*/
	using TestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Avx2>,
		Cfg<double, ::ark::hal::simd::Avx2>
	>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "Vec4UnitTests.h"
