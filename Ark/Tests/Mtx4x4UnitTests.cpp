/*************************************************************************
 * @file
 * @brief Unoptimized 4x4 Mtx Unit Tests
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
#define MTX4X4_TEST_NAMESPACE mtx4x4_unit_tests

/// @brief @includedoc Test/ParameterizedSuitClass.txt
#define MTX4X4_TEST_CLASS Mtx4x4UnitTest


namespace ark::math::test::MTX4X4_TEST_NAMESPACE
{
	/*********************************************************************
	* @brief @includedoc Test/ParameterizedSuitTypesBrief.txt
	* @details @includedoc Test/ParameterizedSuitTypesDetails.txt
	*/
	using TestTypes = ::testing::Types
		<
		Cfg<float, ::ark::hal::simd::None>
		>;
}


//************************************************************************
//  Test Suite
//************************************************************************
#include "Mtx4x4UnitTests.h"
