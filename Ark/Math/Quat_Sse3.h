/*========================================================================
Description
	Optimized specializations of Quat for CPUs with SSE 3 instructions.
	Intel added a handful of instructions to SSE2 which help implement a 
	few basic operations: dot product.

	The signficant added single-precision instructions are:

		__m128 _mm_addsub_ps (__m128 a, __m128 b) => a0-b0, a1+b1, a2-b2, a3+b3
		__m128 _mm_hadd_ps (__m128 a, __m128 b) => a1+a0, a3+a2, b1+b0, b3+b2
		__m128 _mm_hsub_ps (__m128 a, __m128 b) => a0-a1, a2-a3, b0-b1, b2-b3
		__m128 _mm_movehdup_ps (__m128 a) => a1, a1, a3, a3
		__m128 _mm_moveldup_ps (__m128 a) => a0, a0, a2, a2

	The significant added double-precision instructions are:

		__m128d _mm_addsub_pd (__m128d a, __m128d b) => a0-b0, a1+b1
		__m128d _mm_hadd_pd (__m128d a, __m128d b) => a1+a0, b1+b0
		__m128d _mm_hsub_pd (__m128d a, __m128d b) => a0-a1, b0-b1
		__m128d _mm_movedup_pd (__m128d a) => a0, a0

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_SSE3_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE3_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include "Quat_Sse2.h"


/*========================================================================
  Code
========================================================================*/
namespace ark::math
{
	/*====================================================================
	  Classes
	====================================================================*/

	/*--------------------------------------------------------------------
	  SSE 3 Quat<float> is structurally-identical to SSE 1
	--------------------------------------------------------------------*/
	template<> class Quat<float, ark::hal::simd::Sse3> : public Quat<float, ark::hal::simd::Sse2>
	{
		using Quat<float, ark::hal::simd::Sse2>::Quat;
	};


	/*--------------------------------------------------------------------
	  SSE 3 Quat<double> is structurally-identical to SSE 1
	--------------------------------------------------------------------*/
	template<> class Quat<double, ark::hal::simd::Sse3> : public Quat<double, ark::hal::simd::Sse2>
	{
		using Quat<double, ark::hal::simd::Sse2>::Quat;

	public:
		Quat(const Quat<double, ark::hal::simd::Sse2>& q)
			: Quat(q.SseWx(), q.SseYz())
		{}
	};


	/*====================================================================
	  Functions
	====================================================================*/

	/*--------------------------------------------------------------------
		Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse3 SIMD>
	inline auto Dot(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> float
	{
		__m128 squares = _mm_mul_ps(lhs.SseVal(), rhs.SseVal()); // w^2, x^2, y^2, z^2
		__m128 add1st = _mm_hadd_ps(squares, squares);           // w^2+x^2, y^2+z^2, w^2+x^2, y^2+z^2
		__m128 add2nd = _mm_hadd_ps(add1st, add1st);             // w^2+x^2+y^2+z^2, ...
		float result = _mm_cvtss_f32(add2nd);
		return result;
	}


	/*--------------------------------------------------------------------
		Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse3 SIMD>
	auto operator*(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		// Gather data
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();

		// Compute first partial result column
		__m128 l_w = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 0, 0, 0)); // lw, lw, lw, lw
		__m128 a_w = _mm_mul_ps(l_w, r); // lw*rw, lw*rx, lw*ry, lw*rz

		// Compute second partial result column
		__m128 l_x = _mm_shuffle_ps(l, l, _MM_SHUFFLE(1, 1, 1, 1)); // lx, lx, lx, lx
		__m128 r_b = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)); // rx, rw, rz, ry
		__m128 a_x = _mm_mul_ps(l_x, r_b); // lx*rx, lx*rw, lx*rz, lx*ry

		// Compute third partial result column
		__m128 l_y = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 2, 2, 2)); // ly, ly, ly, ly
		__m128 r_c = _mm_shuffle_ps(r_b, r_b, _MM_SHUFFLE(0, 1, 2, 3)); // ry, rz, rw, rx
		__m128 r_k = _mm_set_ps(-0.0, 0.0, 0.0, -0.0); // -, +, +, -
		__m128 r_u = _mm_xor_ps(r_c, r_k); // -ry, rz, rw, -rx
		__m128 a_y = _mm_mul_ps(l_y, r_u); // -ly*ry, ly*rz, ly*rw, -ly*rx

		// Compute fourth partial result column
		__m128 l_z = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 3, 3, 3)); // lz, lz, lz, lz
		__m128 r_d = _mm_shuffle_ps(r_c, r_c, _MM_SHUFFLE(2, 3, 0, 1)); // rz, ry, rx, rw
		__m128 r_l = _mm_shuffle_ps(r_k, r_k, _MM_SHUFFLE(1, 1, 0, 0)); // -, -, +, +
		__m128 r_v = _mm_xor_ps(r_d, r_l); // -rz, -ry, rx, rw
		__m128 a_z = _mm_mul_ps(l_z, r_v); // -lz*rz, -lz*ry, -lz*rx, -lz*rw

		// Add together partial results
		__m128 a_1 = _mm_addsub_ps(a_w, a_x);
		__m128 a_2 = _mm_add_ps(a_1, a_y);
		__m128 a = _mm_add_ps(a_2, a_z);

		return a;
	}


	/*--------------------------------------------------------------------
		Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse3 SIMD>
	inline auto Dot(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> double
	{
		__m128d w2x2 = _mm_mul_pd(lhs.SseWx(), rhs.SseWx()); // w^2, x^2
		__m128d y2z2 = _mm_mul_pd(lhs.SseYz(), rhs.SseYz()); // y^2, z^2
		__m128d add1 = _mm_hadd_pd(w2x2, y2z2);              // w^2+x^2, y^2+z^2
		__m128d add2 = _mm_hadd_pd(add1, add1);              // w^2+x^2+y^2+z^2, ...

		float result = _mm_cvtsd_f64(add2);
		return result;
	}


	/*--------------------------------------------------------------------
	  Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse3 SIMD>
	auto operator*(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		// Gather data
		__m128d n1     = _mm_set_pd(-0.0, 0.0); // Negate element 1
		__m128d lwx    = lhs.SseWx();
		__m128d lyz    = lhs.SseYz();
		__m128d rwx    = rhs.SseWx();
		__m128d ryz    = rhs.SseYz();

		__m128d rxw    = _mm_shuffle_pd(rwx, rwx, _MM_SHUFFLE2(0, 1));
		__m128d rzy    = _mm_shuffle_pd(ryz, ryz, _MM_SHUFFLE2(0, 1));

		__m128d lw     = _mm_unpacklo_pd(lwx, lwx);
		__m128d lx     = _mm_unpackhi_pd(lwx, lwx);
		__m128d ly     = _mm_unpacklo_pd(lyz, lyz);
		__m128d lz     = _mm_unpackhi_pd(lyz, lyz);

		// Compute w & x components
		__m128d awx0   = _mm_mul_pd(lw, rwx);
		__m128d awx1   = _mm_mul_pd(lx, rxw);
		__m128d awx2   = _mm_mul_pd(ly, ryz);
		__m128d awx3   = _mm_mul_pd(lz, rzy);

		__m128d awx01  = _mm_addsub_pd(awx0, awx1);
		__m128d awx012 = _mm_addsub_pd(awx01, awx2);
		__m128d wx     = _mm_sub_pd(awx012, awx3);

		// Compute y & z components
		__m128d ayz0   = _mm_mul_pd(lw, ryz);
		__m128d ayz1   = _mm_mul_pd(lx, rzy);

		__m128d ayz2r  = _mm_xor_pd(rwx, n1);
		__m128d ayz2   = _mm_mul_pd(ly, ayz2r);

		__m128d ayz3   = _mm_mul_pd(lz, rxw);

		__m128d ayz01  = _mm_addsub_pd(ayz0, ayz1);
		__m128d ayz012 = _mm_add_pd(ayz01, ayz2);
		__m128d yz     = _mm_add_pd(ayz012, ayz3);

		return {wx, yz};
	}
}


//========================================================================
#endif
