/*************************************************************************
 * @file
 * @brief Quat WebAssembly 128-bit SIMD-optimized Implementation Unit
 * Tests
 *
 * @details Unit tests for the WebAssembly SIMD 128-bit-optimized Quat
 * class implementations. As with other Quat implementations, this file
 * contains some configuration information used by QuatUnitTests.h, the
 * file containing the actual tests written in a class-independent
 * fashion relying upon preprocessor parameters.
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
#define QUAT_TEST_NAMESPACE quat_wasm128_unit_tests
#define QUAT_TEST_CLASS QuatWasm128UnitTest

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	using QuatTestTypes = ::testing::Types
	<
		Cfg<float, ::ark::hal::simd::Wasm128>
	>;
}

#define MULTIPLICATION_UNSPECIALIZED


//************************************************************************
//  Test Suite
//************************************************************************
#include "QuatUnitTests.h"
