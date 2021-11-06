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

namespace ark::math::test::quat_unit_tests
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
	template<class S, class I>
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
