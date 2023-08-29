 /*************************************************************************
  * @file
  * @brief SSE3-optimized Vec Implementation
  *
  * @details This file defines optimizations to the Vec type family
  * utilizing the SSE3 SIMD ISA. The spec does not define a new register
  * format necessitating a change to the data formats. It does, however, 
  * introduce new instructions enabling more-efficient implementations of 
  * a few algorithms for both single-precision and double-precision 
  * floating-point vectors.
  *
  * @sa Vec.h
  * @se Vec_Sse2.h
  * @sa ::ark::hal::simd::Sse3
  *
  * @author Noah Stein
  * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
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
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 *
	 * @details Concept for declaring parameters of SSE3-optimized
	 * functions.It ensures the parameter is of a type optimized for the
	 * SSE3 SIMD ISA but not earlier SSE specifications.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 *
	 * @sa ark::math::VecSse2
	 * @sa ark::hal::simd::Sse3Family
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecSse3 = VecSse2<V, S, N> &&
		::ark::hal::simd::Sse3Family<typename V::Revision>;

	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized 4-D Single-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE2 to SSE3 do not result in a 
	 * new register specification applicable to further 4-D 
	 * single-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse3Family
	 */
	class VecFloat4Sse3 : public VecFloat4Sse2
	{
	public:
		using Revision = ark::hal::simd::Sse3;
		using VecFloat4Sse2::VecFloat4Sse2;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Sse3> with
	 * VecFloat4Sse3.
	 *
	 * @sa Vec
	 * @sa VecFloat4Sse3
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Sse3>
	{
		using type = VecFloat4Sse3;
	};


	/*********************************************************************
	 * @brief SSE3-optimized 2-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE2 to SSE3 do not result in a
	 * new register specification applicable to further 2-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse3Family
	 */
	class VecDouble2Sse3 : public VecDouble2Sse2
	{
	public:
		using Revision = ark::hal::simd::Sse3;
		using VecDouble2Sse2::VecDouble2Sse2;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 2, Sse3> with
	 * VecDouble2Sse3.
	 *
	 * @sa Vec
	 * @sa VecDouble2Sse3
	 */
	template<>
	struct VectorSelector<double, 2, ark::hal::simd::Sse3>
	{
		typedef VecDouble2Sse3 type;
	};


	/*********************************************************************
	 * @brief SSE3-optimized 4-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE2 to SSE3 do not result in a
	 * new register specification applicable to further 4-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse3Family
	 */
	class VecDouble4Sse3 : public VecDouble4Sse2
	{
	public:
		using Revision = ark::hal::simd::Sse3;
		using VecDouble4Sse2::VecDouble4Sse2;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 4, Sse3> with
	 * VecDouble4Sse3.
	 *
	 * @sa Vec
	 * @sa VecDouble4Sse3
	 */
	template<>
	struct VectorSelector<double, 4, ark::hal::simd::Sse3>
	{
		typedef VecDouble4Sse3 type;
	};


	//====================================================================
	//  2-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE3-optimized 2-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct2D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse3<double, 2> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d m = _mm_mul_pd(lhs.SseVal(), rhs.SseVal()); // lxrx, lyry
		__m128d a = _mm_hadd_pd(m, m); // lxrx+lyry, lxrx+lyry

		double result = _mm_cvtsd_f64(a);
		return result;
	}


	/*********************************************************************
	 * @brief SSE3-optimized 2-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct2D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse3<double, 2> V>
	inline auto Cross(V lhs, V rhs) noexcept -> double
	{
		__m128d l01 = lhs.SseVal();
		__m128d r01 = rhs.SseVal();

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
	 * @brief SSE3-optimized 4-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse3<double, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d v01 = _mm_mul_pd(lhs.Sse01(), rhs.Sse01());
		__m128d va = _mm_hadd_pd(v01, v01);

		__m128d v23 = _mm_mul_pd(lhs.Sse23(), rhs.Sse23());
		__m128d vb = _mm_hadd_pd(v23, v23);

		__m128d dp = _mm_add_pd(va, vb);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//========================================================================
#endif
