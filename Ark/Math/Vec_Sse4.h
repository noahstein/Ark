/*************************************************************************
 * @file
 * @brief Vec<S, N, SIMD> optimizations for the SSE4 ISA.
 * 
 * @details This file defines opitimizations to the basic Vec class for 
 * use on platforms with CPUs possessing SSE4 registers and instructions.
 * The SSE4 spec (both 4.1 and 4.2) continues the register set of SSE3 
 * and adds some additional instructions.
 * 
 * @sa Vec.h
 * @sa Vec_Sse.h
 * @sa Vec_Sse2.h
 * @sa Vec_Sse3.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_SSE4_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_SSE4_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <concepts>

#include "Vec_Sse3.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized 2-D Vec Double Specialization
	 * 
	 * @details As the SSE4 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for double-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE3 class is 
	 * as-is for SSE4.
	 * 
	 * @sa Vec<float, ark::hal::simd::Sse3>
	 ********************************************************************/
	template<>
	class Vec<double, 2, ark::hal::simd::Sse4>
		:	public Vec<double, 2, ark::hal::simd::Sse3>
	{
		using Vec<double, 2, ark::hal::simd::Sse3>::Vec;
	};


	/*********************************************************************
	 * @brief SSE4-optimized 4-D Vec Float Specialization
	 * 
	 * @details As the SSE4 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for single-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE3 class is 
	 * as-is for SSE4.
	 * 
	 * @sa Vec<float, ark::hal::simd::Sse3>
	 ********************************************************************/
	template<>
	class Vec<float, 4, ark::hal::simd::Sse4>
		:	public Vec<float, 4, ark::hal::simd::Sse3>
	{
		using Vec<float, 4, ark::hal::simd::Sse3>::Vec;
	};


	/*********************************************************************
	 * @brief SSE4-optimized 4-D Vec Double Specialization
	 * 
	 * @details As the SSE4 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for double-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the SSE3 class is 
	 * as-is for SSE4.
	 * 
	 * @sa Vec<float, ark::hal::simd::Sse3>
	 ********************************************************************/
	template<>
	class Vec<double, 4, ark::hal::simd::Sse4>
		:	public Vec<double, 4, ark::hal::simd::Sse3>
	{
		using Vec<double, 4, ark::hal::simd::Sse3>::Vec;
	};


	//====================================================================
	//  2-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized Vec<double, 2> Dot Product
	 * 
	 * @details Compute an SSE4-optimized dot product of two 
	 * Vec<double, 2> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 2, ark::hal::simd::Sse4> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 2>.
	 * 
	 * The new intrinsic `_mm_dp_pd` simplifies the function.
	 * 
	 * @include{doc} Math/Vector/DotProduct2D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse4 SIMD>
	inline auto Dot(Vec<double, 2, SIMD> vl, Vec<double, 2, SIMD> vr) -> double
	{
		__m128d dp = _mm_dp_pd(vl.SseVal(), vr.SseVal(), 0x33);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}


	//====================================================================
	//  4-D Vec Float Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized Vec<float, 4> Dot Product
	 * 
	 * @details Compute an SSE4-optimized dot product of two 
	 * Vec<float, 4> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vecfloat, 4, ark::hal::simd::Sse4> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * The new intrinsic `_mm_dp_ps` simplifies the function.
	 * 
	 * @include{doc} Math/Vector/DotProduct2D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse4 SIMD>
	inline auto Dot(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> double
	{
		__m128 dp = _mm_dp_ps(vl.SseVal(), vr.SseVal(), 0xff);
		double result = _mm_cvtss_f32(dp);
		return result;
	}


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized Vec<double, 4> Dot Product
	 * 
	 * @details Compute an SSE4-optimized dot product of two 
	 * Vec<double, 4> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse4> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * The new intrinsic `_mm_dp_pd` permits the simplification.
	 * 
	 * @include{doc} Math/Vector/DotProduct4D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse4 SIMD>
	inline auto Dot(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> double
	{
		__m128d dp1 = _mm_dp_pd(vl.Sse01(), vr.Sse01(), 0x33);
		__m128d dp2 = _mm_dp_pd(vl.Sse23(), vr.Sse23(), 0x33);
		__m128d dp = _mm_add_pd(dp1, dp2);
		
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//========================================================================
#endif
