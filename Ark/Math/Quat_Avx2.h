/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the AVX2 ISA.
 * 
 * @details This file contains specializations of Quat algorithms for the 
 * AVX2 ISA. The major change from AVX to AVX2 for quaternions is the
 * addition of fused multiply-add instructions. These instructions permit 
 * combining what were previously two operations.
 * 
 * @sa ark::math::Quaternion
 * @sa Quat.h
 * @sa Quat_Sse.h
 * @sa Quat_Sse2.h
 * @sa Quat_Sse3.h
 * @sa Quat_Sse4.h
 * @sa Quat_Avx.h
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_AVX2_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_AVX2_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Quat_Avx.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief AVX-optimized Single-precision Floating-point Quaternion
	 * 
	 * @details The AVX2 specification does not define structural changes  
	 * regarding single-precision floating-point vectors; therefore, the 
	 * prior generation is used as-is.
	 *
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 *
	 * @subsubsection Operations
	 * @par
	 * <table>
	 *   <tr><th>Operation<th>Function
	 *   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatAvx2, QuatFloatAvx2)"
	 * </table>
	 ********************************************************************/
	class QuatFloatAvx2 : public QuatFloatAvx
	{
		using QuatFloatAvx::QuatFloatAvx;
	};


	/**
	 * @brief Specialize Quat<float, Sse> with QuatFloatAvx2
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Avx2>
	{
		typedef QuatFloatAvx2 type;
	};


	/*********************************************************************
	 * @brief AVX-optimized Quat Double Specialization
	 * 
	 * @details The AVX2 specification does not define structural changes  
	 * regarding double-precision floating-point vectors; therefore, the 
	 * prior generation is used as-is.
	 *
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 *
	 * @subsubsection Operations
	 * @par
	 * <table>
	 *   <tr><th>Operation<th>Function
	 *   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	 *     <td>@ref "operator*(QuatDoubleAvx2, QuatDoubleAvx2)"
	 * </table>
	 ********************************************************************/
	class QuatDoubleAvx2 : public QuatDoubleAvx
	{
		using QuatDoubleAvx::QuatDoubleAvx;
	};

	/**
	 * @brief Specialize Quat<double, Sse> with QuatDoubleAvx2
	 */

	template<>
	struct QuaternionSelector<double, ark::hal::simd::Avx2>
	{
		typedef QuatDoubleAvx2 type;
	};


	/*********************************************************************
	 * @brief AVX2-optimized Single-precision Multiplication
	 * 
	 * @details Compute the product of two single-precision floating-point 
	 * quaternions using an AVX2-optimized algorithm. This supersedes the
	 * AVX implementation as the new AVX2 fused multiply-add instructions 
	 * permit a more-optimized implementation.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @supersedes{QuatFloatAvx2, operator*(QuatFloatAvx\, QuatFloatAvx)}
	 ********************************************************************/
	auto operator*(QuatFloatAvx2 lhs, QuatFloatAvx2 rhs) -> QuatFloatAvx2
	{
		// Gather data
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();

		// Compute partial result first column
		__m128 l_w = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 0, 0, 0)); // lw, lx, ly, lz
		__m128 a_w = _mm_mul_ps(l_w, r); // lw*rw, lw*rx, lw*ry, lw*rz

		// Compute partial result second column
		__m128 l_x = _mm_shuffle_ps(l, l, _MM_SHUFFLE(1, 1, 1, 1)); // lx, lx, lx, lx
		__m128 r_b = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)); // rx, rw, rz, ry
		__m128 r_j = _mm_set_ps(0.0, -0.0, 0.0, -0.0); // -, +, -, +
		__m128 r_t = _mm_xor_ps(r_b, r_j); // -rx, rw, -rz, ry
		__m128 ps01 = _mm_fmadd_ps(l_x, r_t, a_w); // [a_w +] -lx*rx, -lx*rw, -lx*rz, lx*ry

		// Compute partial result third column
		__m128 l_y = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 2, 2, 2)); // ly, ly, ly, ly
		__m128 r_c = _mm_shuffle_ps(r_b, r_b, _MM_SHUFFLE(0, 1, 2, 3)); // ry, rz, rw, rx
		__m128 r_k = _mm_shuffle_ps(r_j, r_j, _MM_SHUFFLE(0, 1, 1, 0)); // -, +, +, -
		__m128 r_u = _mm_xor_ps(r_c, r_k); // -ry, rz, rw, -rx
		__m128 ps012 = _mm_fmadd_ps(l_y, r_u, ps01); // [a_w+a_x +] -ly*ry, ly*rz, ly*rw, -ly*rx

		// Compute partial result fourth column
		__m128 l_z = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 3, 3, 3)); // lz, lz, lz, lz
		__m128 r_d = _mm_shuffle_ps(r_c, r_c, _MM_SHUFFLE(2, 3, 0, 1)); // rz, ry, rx, rw
		__m128 r_l = _mm_shuffle_ps(r_k, r_k, _MM_SHUFFLE(1, 1, 0, 0)); // -, -, +, +
		__m128 r_v = _mm_xor_ps(r_d, r_l); // -rz, -ry, rx, rw
		__m128 a = _mm_fmadd_ps(l_z, r_v, ps012); // [a_w+a_x+a_y +] -lz*rz, -lz*ry, lz*rx, lz*rw

		return a;
	}

	
	/*********************************************************************
	 * @brief AVX2-optimized Double-precision Multiplication
	 *
	 * @details Compute the product of two double-precision floating-point
	 * quaternions using an AVX2-optimized algorithm. This supersedes the
	 * AVX implementation as the new AVX2 fused multiply-add instructions
	 * permit a more-optimized implementation.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 *
	 * @supersedes{QuatDoubleAvx2, operator*(QuatDoubleAvx\, QuatDoubleAvx)}
	 ********************************************************************/
	auto operator*(QuatDoubleAvx2 lhs, QuatDoubleAvx2 rhs) -> QuatDoubleAvx2
	{
		// Gather data
		__m256d l      = lhs.AvxVal();
		__m256d r      = rhs.AvxVal();

		__m256d lw_lx  = _mm256_permute2f128_pd(l , l, 0);    // lw, lx, lw, lx
		__m256d lw     = _mm256_permute_pd(lw_lx, 0);         // lw, lw, lw, lw
		__m256d lx     = _mm256_permute_pd(lw_lx, 0xF);       // lx, lx, lx, lx

		__m256d ly_lz  = _mm256_permute2f128_pd (l , l, 17);  // ly, lz, ly, lz
		__m256d ly     = _mm256_permute_pd(ly_lz, 0);         // ly, ly, ly, ly
		__m256d lz     = _mm256_permute_pd(ly_lz, 0xF);       // lz, lz, lz, lz

		// Compute partial sum second column
		__m256d r_xwzy = _mm256_permute_pd(r, 5);             // rx, rw, rz, ry
		__m256d ps1    = _mm256_mul_pd(lx, r_xwzy);           // lx*rx, lx*rw, lx*rz, lx*ry

		// Compute partial sum first column
		__m256d ps01    = _mm256_fmaddsub_pd(lw, r, ps1);     // lw*rw, lx*rx, ly*ry, lz*rz

		// Compute partial sum third column
		__m256d r_yzwx = _mm256_permute2f128_pd(r, r, 1);     // ry, rz, rw, rx
		__m256d n2     = _mm256_set_pd(-0.0, 0.0, 0.0, -0.0); // -, +, +, -
		__m256d r_2n   = _mm256_xor_pd(r_yzwx, n2);           // -ry, rz, rw, -rx
		__m256d ps012  = _mm256_fmadd_pd(ly, r_2n, ps01);     // -ly*ry, ly*rz, ly*rw, -ly*rx

		// Compute partial sum fourth column
		__m256d r_zyxw = _mm256_permute_pd(r_yzwx, 5);        // rz, ry, rx, rw
		__m256d n3     = _mm256_permute_pd(n2, 0);            // -, -, +, +
		__m256d r_3n   = _mm256_xor_pd(r_zyxw, n3);           // -rx, -ry, rx, rw
		__m256d a      = _mm256_fmadd_pd(lz, r_3n, ps012);    // -lz*rx, -lz*ry, lz*rx, lz*rw

		return a;
	}
}


//************************************************************************
#endif
