/*************************************************************************
 * @file
 * @brief SSE4-optimized Vec Implementation
 *
 * @details This file defines optimizations to the Vec type family
 * utilizing the SSE4 SIMD ISA. As with SSE3, this spec does not define a 
 * new register form necessitating a new data class. The new ISA includes 
 * instructions to further optimize the dot product.
 *
 * @sa Vec.h
 * @se Vec_Sse3.h
 * @sa ::ark::hal::simd::Sse4
 *
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
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
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 *
	 * @details Concept for declaring parameters of SSE4-optimized
	 * functions.It ensures the parameter is of a type optimized for the
	 * SSE4 SIMD ISA but not earlier SSE specifications.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 *
	 * @sa ark::math::VecSse2=3
	 * @sa ark::hal::simd::Sse4Family
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecSse4 = VecSse3<V, S, N> &&
		::ark::hal::simd::Sse4Family<typename V::Revision>;


	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized 4-D Single-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE3 to SSE4 do not result in a
	 * new register specification applicable to further 4-D
	 * single-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse4Family
	 */
	class VecFloat4Sse4 : public VecFloat4Sse3
	{
	public:
		using Revision = ark::hal::simd::Sse4;
		using VecFloat4Sse3::VecFloat4Sse3;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Sse4> with
	 * VecFloat4Sse4.
	 *
	 * @sa Vec
	 * @sa VecFloat4Sse4
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Sse4>
	{
		using type = VecFloat4Sse4;
	};


	/*********************************************************************
	 * @brief SSE4-optimized 2-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE3 to SSE4 do not result in a
	 * new register specification applicable to further 2-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse4Family
	 */
	class VecDouble2Sse4 : public VecDouble2Sse3
	{
	public:
		using Revision = ark::hal::simd::Sse4;
		using VecDouble2Sse3::VecDouble2Sse3;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 2, Sse4> with
	 * VecDouble2Sse4.
	 *
	 * @sa Vec
	 * @sa VecDouble2Sse4
	 */
	template<>
	struct VectorSelector<double, 2, ark::hal::simd::Sse4>
	{
		typedef VecDouble2Sse4 type;
	};


	/*********************************************************************
	 * @brief SSE4-optimized 4-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE3 to SSE4 do not result in a
	 * new register specification applicable to further 4-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse4Family
	 */
	class VecDouble4Sse4 : public VecDouble4Sse3
	{
	public:
		using Revision = ark::hal::simd::Sse4;
		using VecDouble4Sse3::VecDouble4Sse3;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 4, Sse4> with
	 * VecDouble4Sse4.
	 *
	 * @sa Vec
	 * @sa VecDouble4Sse4
	 */
	template<>
	struct VectorSelector<double, 4, ark::hal::simd::Sse4>
	{
		using type = VecDouble4Sse4;
	};


	//====================================================================
	//  2-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized 2-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct2D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse4<double, 2> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d dp = _mm_dp_pd(lhs.SseVal(), rhs.SseVal(), 0x33);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}


	//====================================================================
	//  4-D Vec Float Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized 4-D Single-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse4<float, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128 dp = _mm_dp_ps(lhs.SseVal(), rhs.SseVal(), 0xff);
		double result = _mm_cvtss_f32(dp);
		return result;
	}


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE4-optimized 4-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse4<double, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d dp1 = _mm_dp_pd(lhs.Sse01(), rhs.Sse01(), 0x33);
		__m128d dp2 = _mm_dp_pd(lhs.Sse23(), rhs.Sse23(), 0x33);
		__m128d dp = _mm_add_pd(dp1, dp2);
		
		double result = _mm_cvtsd_f64(dp);
		return result;
	}
}


//========================================================================
#endif
