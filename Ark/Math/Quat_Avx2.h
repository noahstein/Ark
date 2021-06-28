/*========================================================================
Description
  Optimized specializations of Quat for CPUs with AVX2 instructions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_AVX2_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_AVX2_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include "Quat_Avx.h"


/*========================================================================
  Code
========================================================================*/
namespace ark::math
{
	/*====================================================================
	  Classes
	====================================================================*/

	/*--------------------------------------------------------------------
	  AVX2 Quat<float> is structurally-identical to that of AVX
	--------------------------------------------------------------------*/
	template<> class Quat<float, ark::hal::simd::Avx2> : public Quat<float, ark::hal::simd::Avx>
	{
		using Quat<float, ark::hal::simd::Avx>::Quat;
	};


	/*--------------------------------------------------------------------
	  AVX2 Quat<double> is structurally-identical to that of AVX
	--------------------------------------------------------------------*/
	template<> class Quat<double, ark::hal::simd::Avx2> : public Quat<double, ark::hal::simd::Avx>
	{
		using Quat<double, ark::hal::simd::Avx>::Quat;
	};


	/*====================================================================
	  Functions
	====================================================================*/

	/*--------------------------------------------------------------------
		Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx2 SIMD>
	auto operator*(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
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


//========================================================================
#endif
