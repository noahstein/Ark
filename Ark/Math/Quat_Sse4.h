/*========================================================================
Description
  Optimized specializations of Quat for CPUs with SSE4 instructions.

  The new significant float instructions:

      __m128 _mm_blend_ps (__m128 a, __m128 b, const int imm8) -> a0|b0, a1|b1, a2|b2, a3|b3
      __m128 _mm_blendv_ps (__m128 a, __m128 b, __m128 mask) -> a0|b0, a1|b1, a2|b2, a3|b3
      __m128 _mm_dp_ps (__m128 a, __m128 b, const int imm8) -> a0*b0? + a1*b1? + a2*b2? + a3*b3?, 0, 0, 0
      int _mm_extract_ps (__m128 a, const int imm8) -> a[0|1|2|3]
      __m128 _mm_insert_ps (__m128 a, __m128 b, const int imm8) -> a w/ one element from b anywhere or 0

  The new significant double instructions:

      __m128d _mm_blend_pd (__m128d a, __m128d b, const int imm8) -> a0|b0, a1|b1
      __m128d _mm_blendv_pd (__m128d a, __m128d b, __m128d mask) -> a0|b0, a1|b1
      __m128d _mm_dp_pd (__m128d a, __m128d b, const int imm8) -> a0*b0?+a1*b1?|0, ...

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_SSE4_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE4_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include "Quat_Sse3.h"


/*========================================================================
  Code
========================================================================*/
namespace ark::math
{
	/*====================================================================
	  Classes
	====================================================================*/

	/*--------------------------------------------------------------------
	  SSE4 Quat<float> is structurally-identical to SSE 1-3
	--------------------------------------------------------------------*/
	template<> class Quat<float, ark::hal::simd::Sse4> : public Quat<float, ark::hal::simd::Sse3>
	{
		using Quat<float, ark::hal::simd::Sse3>::Quat;
	};


	/*--------------------------------------------------------------------
	  SSE4 Quat<double> is structurally-identical to SSE 1
	--------------------------------------------------------------------*/
	template<> class Quat<double, ark::hal::simd::Sse4> : public Quat<double, ark::hal::simd::Sse3>
	{
		using Quat<double, ark::hal::simd::Sse3>::Quat;
	};


	/*====================================================================
	  Functions
	====================================================================*/

	/*--------------------------------------------------------------------
		Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse4 SIMD>
	inline auto Dot(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> float
	{
		__m128 dp = _mm_dp_ps(lhs.SseVal(), rhs.SseVal(), 0xFF);
		float result = _mm_cvtss_f32(dp);
		return result;
	}


	/*--------------------------------------------------------------------
		Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse4 SIMD>
	auto operator*(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		// Gather data
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();
		__m128 n = _mm_set1_ps(-0.0);
		__m128 z = _mm_setzero_ps();

		// w = lw*rw - lx*rx - ly*ry - lz*rz
		__m128 r_wm   = _mm_blend_ps(z, n, 0b1110); // +, -, -, -
		__m128 r_w    = _mm_xor_ps(r, r_wm);        // rw, -rx, -ry, -rz
		__m128 w      = _mm_dp_ps(l, r_w, 0xF1);    // lw*rw - lx*rx - ly*ry - lz*rz, 0, 0, 0

		// x = lw*rx + lx*rw + ly*rz - lz*ry
		__m128 r_xm   = _mm_blend_ps(z, n, 0b1000); // +, +, +, -
		__m128 r_xwzy = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)); // rx, rw, rz, ry
		__m128 r_xn   = _mm_xor_ps(r_xwzy, r_xm); // rx, rw, rz, -ry
		__m128 x      = _mm_dp_ps(l, r_xn, 0xF2); // 0, lw*rx + lx*rw + ly*rz - lz*ry, 0, 0

		// y = lw*ry - lx*rz + ly*rw + lz*rx
		__m128 r_ym   = _mm_blend_ps(z, n, 0b0010); // +, -, +, +
		__m128 r_yzwx = _mm_shuffle_ps(r_xwzy, r_xwzy, _MM_SHUFFLE(0, 1, 2, 3)); // ry, rz, rw, rx
		__m128 r_yn   = _mm_xor_ps(r_yzwx, r_ym); // ry, -rz, rw, rx
		__m128 y      = _mm_dp_ps(l, r_yn, 0xF4); // 0, 0, lw*ry - lx*rz + ly*rw + lz*rx, 0

		// z = lw*rz + lx*ry - ly*rx + lz*rw
		__m128 r_zm   = _mm_blend_ps(z, n, 0b0100); // +, +, -, +
		__m128 r_zyxw = _mm_shuffle_ps(r_yzwx, r_yzwx, _MM_SHUFFLE(2, 3, 0, 1)); // rz, ry, rx, rw
		__m128 r_zn   = _mm_xor_ps(r_zyxw, r_zm); // rz, ry, -rx, rw
		__m128 zz     = _mm_dp_ps(l, r_zn, 0xFF); // 0, 0, 0, lw*rz + lx*ry - ly*rx + lz*rw

		// Assemble final result
		__m128 a_wx  = _mm_blend_ps(w, x, 0b0010);
		__m128 a_wxy = _mm_blend_ps(a_wx, y, 0b0100);
		__m128 a     = _mm_blend_ps(a_wxy, zz, 0b1000);

		return a;
	}


	/*--------------------------------------------------------------------
		Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse4 SIMD>
	inline auto Dot(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> double
	{
		__m128d dp_wx = _mm_dp_pd(lhs.SseWx(), rhs.SseWx(), 0xFF);
		__m128d dp_yz = _mm_dp_pd(lhs.SseYz(), rhs.SseYz(), 0xFF);
		__m128d dp = _mm_add_pd(dp_wx, dp_yz);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//========================================================================
#endif
