/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the SSE4 ISA.
 * 
 * @details This file contains optimiztions of Quat algorithms specific 
 * to the SSE4 ISA. The 4th generation of the SSE spec contains no new 
 * register formats; however, it introduces new instructions that enable 
 * some algorithms to be better optimized. Consequently, only those 
 * algorithms are reimplemented here. Those from earlier generations that 
 * cannot be improved upon will be inherited automatically.
 * 
 * @sa ark::math::Quaternion
 * @sa Quat.h
 * @sa Quat_Sse.h
 * @sa Quat_Sse2.h
 * @sa Quat_Sse3.h
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_SSE4_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE4_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Quat_Sse3.h"

class QuatFloatAvx2;


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	* @brief SSE4-optimized Single-precision Floating-point Quaternion
	* 
	* @details The SSE4 speicificaiton makes no register changes that 
	* affect the data layout of float quaternions, so the SSE3 data 
	* format is reused. SSE4 does, however, introduce a dot product 
	* instruction permitting more-efficient implementations of the dot 
	* product and multiplication functions.
	*
	* @subsubsection Concepts Concepts
	* @par
	* @ref "ark::math::Quaternion" "Quaternion"
	*
	* @subsubsection Operations
	* @par
	* <table>
	*	<tr><th>Operation<th>Function
	*   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	*     <td>@ref "Dot(QuatFloatSse4, QuatFloatSse4)"
	*   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	*     <td>@ref "operator*(QuatFloatSse4, QuatFloatSse4)"
	* </table>
	*
	* @sa QuatFloatSse3
	********************************************************************/
	class QuatFloatSse4 : public QuatFloatSse3
	{
	public:
		/// Tag for revision this implementation's generation in the SIMD family.
		using Revision = ark::hal::simd::Sse4;

		using QuatFloatSse3::QuatFloatSse3;
	};


	/**
	 * @brief Specialize Quat<float, Sse4> with QuatFloatSse4
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Sse4>
	{
		typedef QuatFloatSse4 type;
	};


	/*********************************************************************
	 * @brief SSE4-optimized Double-precision Floating-point Quaternion
	 * 
	 * @details The SSE4 specification makes no register changes that 
	 * affect the data layout of double quaternions, so the SSE3 data 
	 * format is reused. SSE4 does, however, introduce a dot product 
	 * instruction permitting more-efficient implementations of the dot 
	 * product function.
	 *
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 *
	 * @subsubsection Operations
	 * @par
	 * <table>
	 *   <tr><th>Operation<th>Function
	 *   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	 *     <td>@ref "Dot(QuatDoubleSse4, QuatDoubleSse4)"
	 * </table>
	 *
	 * @sa QuatDoubleSse3
	 ********************************************************************/
	class QuatDoubleSse4 : public QuatDoubleSse3
	{
		using QuatDoubleSse3::QuatDoubleSse3;
	};


	/**
	 * @brief Specialize Quat<double, Sse4> with QuatDoubleSse4
	 */
	template<>
	struct QuaternionSelector<double, ark::hal::simd::Sse4>
	{
		typedef QuatDoubleSse4 type;
	};


	/*********************************************************************
	 * @brief SSE4-optimized Single-precision Quaternion Dot Product
	 * 
	 * @details Compute the dot product of two single-precision 
	 * floating-point quaternions using an SSE4-optimized algorithm.SSE4's 
	 * new dot product instruction permits a simple implementation.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @supersedes{QuatFloatSse4, Dot(QuatFloatSse3\, QuatFloatSse3)}
	 ********************************************************************/
	inline auto Dot(QuatFloatSse4 lhs, QuatFloatSse4 rhs) -> float
	{
		__m128 dp = _mm_dp_ps(lhs.SseVal(), rhs.SseVal(), 0xFF);
		float result = _mm_cvtss_f32(dp);
		return result;
	}


	/*********************************************************************
	 * @brief SSE4-optimized Single-precision Quaternion Multiplication 
	 * 
	 * @details Compute the product of two single-precision floating-point 
	 * quaternions using an SSE3-optimized algorithm. The new dot product 
	 * instruction enabled the new optimization.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 *
	 * @supersedes{QuatFloatSse4, operator*(QuatFloatSse3\, QuatFloatSse3)}
	 ********************************************************************/
	auto operator*(QuatFloatSse4 lhs, QuatFloatSse4 rhs) -> QuatFloatSse4
	{
		// Gather data
		__m128 l  = lhs.SseVal();
		__m128 r  = rhs.SseVal();
		__m128 nz = _mm_set1_ps(-0.0);
		__m128 pz = _mm_setzero_ps();

		// w = lw*rw - lx*rx - ly*ry - lz*rz
		__m128 r_wm   = _mm_blend_ps(pz, nz, 0b1110); // +, -, -, -
		__m128 r_w    = _mm_xor_ps(r, r_wm); // rw, -rx, -ry, -rz
		__m128 w      = _mm_dp_ps(l, r_w, 0xF1); // lw*rw - lx*rx - ly*ry - lz*rz, 0, 0, 0

		// x = lw*rx + lx*rw + ly*rz - lz*ry
		__m128 r_xm   = _mm_blend_ps(pz, nz, 0b1000); // +, +, +, -
		__m128 r_xwzy = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)); // rx, rw, rz, ry
		__m128 r_xn   = _mm_xor_ps(r_xwzy, r_xm); // rx, rw, rz, -ry
		__m128 x      = _mm_dp_ps(l, r_xn, 0xF2); // 0, lw*rx + lx*rw + ly*rz - lz*ry, 0, 0

		// y = lw*ry - lx*rz + ly*rw + lz*rx
		__m128 r_ym   = _mm_blend_ps(pz, nz, 0b0010); // +, -, +, +
		__m128 r_yzwx = _mm_shuffle_ps(r_xwzy, r_xwzy, _MM_SHUFFLE(0, 1, 2, 3)); // ry, rz, rw, rx
		__m128 r_yn   = _mm_xor_ps(r_yzwx, r_ym); // ry, -rz, rw, rx
		__m128 y      = _mm_dp_ps(l, r_yn, 0xF4); // 0, 0, lw*ry - lx*rz + ly*rw + lz*rx, 0

		// z = lw*rz + lx*ry - ly*rx + lz*rw
		__m128 r_zm   = _mm_blend_ps(pz, nz, 0b0100); // +, +, -, +
		__m128 r_zyxw = _mm_shuffle_ps(r_yzwx, r_yzwx, _MM_SHUFFLE(2, 3, 0, 1)); // rz, ry, rx, rw
		__m128 r_zn   = _mm_xor_ps(r_zyxw, r_zm); // rz, ry, -rx, rw
		__m128 z      = _mm_dp_ps(l, r_zn, 0xFF); // 0, 0, 0, lw*rz + lx*ry - ly*rx + lz*rw

		// Assemble final result
		__m128 a_wx  = _mm_blend_ps(w, x, 0b0010);
		__m128 a_wxy = _mm_blend_ps(a_wx, y, 0b0100);
		__m128 a     = _mm_blend_ps(a_wxy, z, 0b1000);

		return a;
	}


	/*********************************************************************
	 * @brief SSE4-optimized Double-precision Quaternion Dot Product
	 *
	 * @details Compute the dot product of two double-precision
	 * floating-point quaternions using an SSE4-optimized algorithm.SSE4's
	 * new dot product instruction permits a simple implementation.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 *
	 * @supersedes{QuatDoubleSse4, Dot(QuatDoubleSse3\, QuatDoubleSse3)}
	 ********************************************************************/
	inline auto Dot(QuatDoubleSse4 lhs, QuatDoubleSse4 rhs) -> double
	{
		__m128d dp_wx = _mm_dp_pd(lhs.SseWx(), rhs.SseWx(), 0xFF);
		__m128d dp_yz = _mm_dp_pd(lhs.SseYz(), rhs.SseYz(), 0xFF);
		__m128d dp = _mm_add_pd(dp_wx, dp_yz);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//************************************************************************
#endif
