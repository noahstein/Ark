/*************************************************************************
 * @file
 * @brief Vec<S, N, SIMD> optimizations for the SSE3 ISA.
 * 
 * @details This file defines opitimizations to the basic Vec class for 
 * use on platforms with CPUs possessing SSE3 registers and instructions.
 * The SSE3 spec continues the register set of SSE2 and, for the purposes 
 * of vectors, adds a few instructions to simplify a few functions and 
 * make them faster.
 * 
 * @sa Vec.h
 * @sa Vec_Sse.h
 * @sa Vec_Sse2.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_SSE3_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_SSE3_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <concepts>

#include "Vec_Sse2.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized 2-D Vec Double Specialization
	 * 
	 * @details As the SSE3 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for double-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE2 class is 
	 * as-is for SSE3.
	 * 
	 * @sa Vec<double, 2, ark::hal::simd::Sse2>
	 ********************************************************************/
	template<>
	class Vec<double, 2, ark::hal::simd::Sse3>
		:	public Vec<double, 2, ark::hal::simd::Sse2>
	{
		using Vec<double, 2, ark::hal::simd::Sse2>::Vec;
	};


	/*********************************************************************
	 * @brief SSE3-optimized 4-D Vec Float Specialization
	 * 
	 * @details As the SSE3 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for single-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE2 class is 
	 * as-is for SSE3.
	 * 
	 * @sa Vec<float, 4, ark::hal::simd::Sse2>
	 ********************************************************** **********/
	template<>
	class Vec<float, 4, ark::hal::simd::Sse3>
		:	public Vec<float, 4, ark::hal::simd::Sse2>
	{
		using Vec<float, 4, ark::hal::simd::Sse2>::Vec;
	};


	/*********************************************************************
	 * @brief SSE3-optimized 4-D Vec Double Specialization
	 * 
	 * @details As the SSE3 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for double-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE2 class is 
	 * as-is for SSE3.
	 * 
	 * @sa Vec<double, 4, ark::hal::simd::Sse2>
	 ********************************************************************/
	template<>
	class Vec<double, 4, ark::hal::simd::Sse3>
		:	public Vec<double, 4, ark::hal::simd::Sse2>
	{
		using Vec<double, 4, ark::hal::simd::Sse2>::Vec;
	};


	//====================================================================
	//  2-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized Vec<double, 2> Dot Product
	 * 
	 * @details Compute an SSE3-optimized dot product of two 
	 * Vec<double, 2> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 2, ark::hal::simd::Sse3> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 2>.
	 * 
	 * The new intrinsic `_mm_hadd_pd` permits removing a shuffle to add 
	 * the two multiplied components.
	 * 
	 * @include{doc} Math/Vector/DotProduct2D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse3 SIMD>
	inline auto Dot(Vec<double, 2, SIMD> vl, Vec<double, 2, SIMD> vr) -> double
	{
		__m128d m = _mm_mul_pd(vl.SseVal(), vr.SseVal()); // lxrx, lyry
		__m128d a = _mm_hadd_pd(m, m); // lxrx+lyry, lxrx+lyry

		double result = _mm_cvtsd_f64(a);
		return result;
	}


	/*********************************************************************
	 * @brief SSE3-optimized Vec<double, 2> Cross Product
	 * 
	 * @details Compute an SSE3-optimized cross product of two 
	 * Vec<double, 2> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 2, ark::hal::simd::Sse3> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 2>.
	 * 
	 * The new intrinsic `_mm_hsub_pd` permits removing a shuffle to 
	 * perform the subtraction of the second term from the first.
	 * 
	 * @include{doc} Math/Vector/CrossProduct2D.txt
	 * 
	 * @sa Cross(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse3 SIMD>
	inline auto Cross(Vec<double, 2, SIMD> vl, Vec<double, 2, SIMD> vr) -> double
	{
		__m128d l01 = vl.SseVal();
		__m128d r01 = vr.SseVal();

		__m128d r10 = _mm_shuffle_pd(r01, r01, _MM_SHUFFLE2(0, 1)); // r1, r0
		__m128d a01 = _mm_mul_pd(l01, r10); // l0r1, l1r0
		__m128d a = _mm_hsub_pd(a01, a01); // l0r1-l1r0, l1r0-l0r1

		double result = _mm_cvtsd_f64(a);
		return result;
	}


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized Vec<double, 4> Dot Product
	 * 
	 * @details Compute an SSE3-optimized dot product of two 
	 * Vec<double, 4> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse3> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * The new intrinsic `_mm_hadd_pd` permits removing a shuffle both 
	 * times two multiplied components in a register are added together.
	 * 
	 * @include{doc} Math/Vector/DotProduct4D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse3 SIMD>
	inline auto Dot(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> double
	{
		__m128d v01 = _mm_mul_pd(vl.Sse01(), vr.Sse01());
		__m128d va = _mm_hadd_pd(v01, v01);

		__m128d v23 = _mm_mul_pd(vl.Sse23(), vr.Sse23());
		__m128d vb = _mm_hadd_pd(v23, v23);

		__m128d dp = _mm_add_pd(va, vb);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//========================================================================
#endif
