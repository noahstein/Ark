/*************************************************************************
 * @file
 * @brief Math Parametric Test Suite Configuration
 * 
 * @details This file contains useful Google Test parametric test 
 * configuration helpers to aid test files in implementing test suites.
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_TESTS_CFG_H_INCLUDE_GUARD)
#define ARK_TESTS_CFG_H_INCLUDE_GUARD

//************************************************************************
//  Dependencies
//************************************************************************
#include "Ark/Hal/Simd/Simd.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math::test
{
	/*********************************************************************
	 * @brief Math Parametric Unit Test Suite Configuration
	 * 
	 * @tparam S The scalar type of the specialization
	 * 
	 * @tparam I The ISA tag of the specialization
	 * 
	 * @details One of the main features of the math module is that it 
	 * is designed as platform-independent implementations that may be 
	 * supplemented as desired by specializations for specific hardware 
	 * architectures. The primary CPU aspect relating to math functions is 
	 * the SIMD architecture.
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
}


//************************************************************************
#endif