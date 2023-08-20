/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the ARM Neon AArch64.
 *
 * @details This file defines optimizations to the basic Quat class for 
 * use on platforms with ARM CPUs possessing registers and instructions 
 * defined in the Neon AArch64 ISA. This file contains a a class and 
 * supporting non-member functions implementing Quaternion math routines 
 * optimized for the original Neon ISA. The class is then specified as a 
 * specialization of the Quat<S, I> class to be used in client code.
 *
 * @sa ark::math::Quaternion
 * @sa Quat.h
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_NEON64_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_NEON64_H_INCLUDE_GUARD

/*************************************************************************
 * Dependencies
 ************************************************************************/
#include "Quat_Neon32.h"


/*************************************************************************
 * Code
 ************************************************************************/
namespace ark::math
{
	/*********************************************************************
	 * @brief Quaternion with ARM Neon AArch64 Optimization
	 *
	 * @details The concept of quaternion class of a specific scalar type 
	 * optimized for the Neon AArch64 ISA. Functions that implement 
	 * algorithms with Neon AArch64 instructions use this concept to 
	 * define their parameters.
	 *
	 * @sa QuaternionNone
	 * @sa ark::hal::simd::Neon64
	 */
	template<typename Q, typename S>
	concept QuatNeon64 = QuaternionNone<Q, S> &&
		::ark::hal::simd::Neon64Family<typename Q::Revision>;


	/*********************************************************************
	 * @brief Single-precision Floating-point Quaternion Optimized for 
	 * the ARM Neon AArch64 ISA.
	 *
	 * @details The Neon64 AArch64 ISA defines new instructions but the 
	 * register format remains the same; therefore, the register format 
	 * is re-used.
	 *
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 */
	class QuatFloatNeon64 : public QuatFloatNeon32
	{
	public:
		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Neon64;

		using QuatFloatNeon32::QuatFloatNeon32;
	};


	/**
	 * @brief Specialize Quat<float, Neon64> with QuatFloatNeon64
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Neon64>
	{
		typedef QuatFloatNeon64 type;
	};


	/*********************************************************************
	 * @brief Neon64-optimized Single-precision Quaternion Dot Product
	 *
	 * @details Compute the dot product of two single-precision
	 * floating-point quaternions using an algorithm optimized using the 
	 * ARM Neon AArch64 ISA. The horizontal add instruction added to 
	 * AArch64 supports a more-efficient implementation.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 *
	 * @supersedes{QuatFloatNeon32, Dot(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon64<float> Q>
	inline auto Dot(Q lhs, Q rhs) noexcept -> float
	{
		float32x4_t muls = vmulq_f32(lhs.NeonVal(), rhs.NeonVal());
		float result = vaddvq_f32(muls);
		return result;
	}


	/*********************************************************************
	 * @brief Neon64-optimized Single-precision Quaternion Multiplication
	 *
	 * @details Compute the product of two single-precision floating-point
	 * quaternions using an ARM Neon AArch64-optimized algorithm.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 *
	 * @supersedes{QuatFloatNeon32,operator*(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon64<float> Q>
	auto operator*(Q lhs, Q rhs) noexcept -> Q
	{
		static float32_t negations[] =
		{
			 1.0f, -1.0f, -1.0f,
			 1.0f, -1.0f,  1.0f,
			-1.0f,  1.0f, -1.0f,
			-1.0f,  1.0f,  1.0f
		};

		// Gather data
		float32x4_t l = lhs.NeonVal();
		float32x4_t r = rhs.NeonVal();
		float32x4x3_t neg = vld3q_f32(negations);

		// Compute first partial result
		float32x4_t r0 = vdupq_laneq_f32(r, 0);
		float32x4_t p0 = vmulq_f32(l, r0);

		// Compute 2nd partial result
		float32x4_t zero = vdupq_n_f32(0.0f);
		float32x4_t r1 = vdupq_laneq_f32(r, 1);
		float32x4_t m1 = vcmlaq_rot90_f32(zero, r1, l);
		float32x4_t p1 = vfmaq_f32(p0, m1, neg.val[0]);

		// Compute 3rd partial result
		float32x4_t r2 = vdupq_laneq_f32(r, 2);
		float32x4_t ex = vextq_f32(l, l, 2);
		float32x4_t m2 = vmulq_f32(ex, r2);
		float32x4_t p2 = vfmaq_f32(p1, m2, neg.val[1]);

		// Compute 4th partial result
		float32x4_t r3 = vdupq_laneq_f32(r, 3);
		float32x4_t l3 = vrev64q_f32(ex);
		float32x4_t m3 = vmulq_f32(l3, r3);
		float32x4_t result = vfmaq_f32(p2, m3, neg.val[2]);

		return result;
	}
}


//************************************************************************
#endif
