/*************************************************************************
 * @file
 * @brief Math Parametric Test Suite Configuration
 * 
 * @details This file contains useful Google Test parametric test 
 * configuration helpers to aid test files in implementing test suites.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "Ark/Hal/Simd/SImd.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math::test
{
	/*********************************************************************
	 * @brief Math Parametric Unit Test Suite Configuration
	 * 
	 * @tparam S The scalar type of the quaternion specialization
	 * 
	 * @tparam I The ISA tag of the quaternion specialization
	 * 
	 * @details One of the main features of the math module is that it 
	 * is architected as platform-independent implementations that may be 
	 * supplemented as desired by specializations for specific hardware 
	 * achitectures. The primary CPU aspect relating to math functions is 
	 * the SIMD arhitecture.
	 ********************************************************************/
	template<class S, class I = ::ark::hal::simd::None>
	struct Cfg
	{
		/**
		 * @brief The scalar type of the quaternion to undergo testing
		 */
		using Scalar = S;

		/**
		 * @brief The ISA tag of the quaternion to undergo testing
		 */
		using Isa = I;
	};



	/*********************************************************************
	 * @brief Set of standard numercial types to test in a suite
	 * 
	 * @details Each template parameter in the list is used as a 
	 * configuration for a separate parametric execution of the suite.
	 ********************************************************************/
	using StdTypes = ::testing::Types
	<	Cfg<int>
	,	Cfg<long>
	,	Cfg<float>
	,	Cfg<double>
	>;

	/*********************************************************************
	 * @brief Set of SSE numerical types to test with the suite
	 * 
	 * @details Each template parameter in the list is used as a 
	 * configuration for a separate parametric execution of the suite.
	 ********************************************************************/
	using SseTypes = ::testing::Types
	<	Cfg<float, ::ark::hal::simd::None>
	,	Cfg<double, ::ark::hal::simd::None>

#if defined(SIMD_HAS_SSE)
	,	Cfg<float, ::ark::hal::simd::Sse>
	,	Cfg<double, ::ark::hal::simd::Sse>
#endif

#if defined(SIMD_HAS_SSE2)
	,	Cfg<float, ::ark::hal::simd::Sse2>
	,	Cfg<double, ::ark::hal::simd::Sse2>
#endif

#if defined(SIMD_HAS_SSE3)
	,	Cfg<float, ::ark::hal::simd::Sse3>
	,	Cfg<double, ::ark::hal::simd::Sse3>
#endif

#if defined(SIMD_HAS_SSE4)
	,	Cfg<float, ::ark::hal::simd::Sse4>
	,	Cfg<double, ::ark::hal::simd::Sse4>
#endif

#if defined(SIMD_HAS_AVX)
	,	Cfg<float, ::ark::hal::simd::Avx>
	,	Cfg<double, ::ark::hal::simd::Avx>
#endif

#if defined(SIMD_HAS_AVX2)
	,	Cfg<float, ::ark::hal::simd::Avx2>
	,	Cfg<double, ::ark::hal::simd::Avx2>
#endif		
	>;
}
