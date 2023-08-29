/*************************************************************************
 * @file
 * @brief AVX2-optimized Vec Implementation
 *
 * @details This file defines optimizations to the Vec type family
 * utilizing the AVX SIMD ISA. The major upgrade from AVX to AVX2 is the 
 * extension of registers from 256 bits to 512. As 8-D vectors aren't 
 * enormously useful, this merely promotes the original AVX types.
 * 
 * @note What is useful is the parallel processing in a structure of 
 * arrays design; therefore, this file could contain a future 
 * implementation of parallel array processing, should it be added.
 *
 * @sa Vec.h
 * @se Vec_Avx.h
 * @sa ::ark::hal::simd::Avx2
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_AVX2_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_AVX2_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <concepts>

#include "Vec_Avx.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief AVX2-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 *
	 * @details Concept for declaring parameters of AVX2-optimized
	 * functions.It ensures the parameter is of a type optimized for the
	 * AVX2 SIMD ISA but not the original AVX ISA and all of the SSE 
	 * specifications.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 *
	 * @sa ark::math::Vec
	 * @sa ark::hal::simd::AVx2Family
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecAvx2 = VecAvx<V, S, N> &&
		::ark::hal::simd::Avx2Family<typename V::Revision>;


	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief AVX2-optimized 4-D Single-precision Floating-point Vector
	 *
	 * @details The changes changes from AVX to AVX2 do not result in a
	 * new register specification applicable to further 4-D
	 * single-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:AvxFamily
	 */
	class VecFloat4Avx2 : public VecFloat4Avx
	{
	public:
		using Revision = ark::hal::simd::Avx2;
		using VecFloat4Avx::VecFloat4Avx;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Avx2> with
	 * VecFloat4Avx2.
	 *
	 * @sa Vec
	 * @sa VecFloat4Avx
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Avx2>
	{
		using type = VecFloat4Avx2;
	};


	/*********************************************************************
	 * @brief AVX2-optimized 2-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from AVX to AVX2 do not result in a
	 * new register specification applicable to further 2-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Avx2Family
	 */
	class VecDouble2Avx2 : public VecDouble2Avx
	{
	public:
		using Revision = ark::hal::simd::Avx2;
		using VecDouble2Avx::VecDouble2Avx;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 2, Avx> with
	 * VecDouble2Avx.
	 *
	 * @sa Vec
	 * @sa VecDouble2Avx
	 */
	template<>
	struct VectorSelector<double, 2, ark::hal::simd::Avx2>
	{
		using type = VecDouble2Avx2;
	};


	/*********************************************************************
	 * @brief AVX2-optimized 4-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from AVX to AVX2 do not result in a
	 * new register specification applicable to further 4-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Avx2Family
	 */
	class VecDouble4Avx2 : public VecDouble4Avx
	{
	public:
		using Revision = ark::hal::simd::Avx2;
		using VecDouble4Avx::VecDouble4Avx;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 4, Avx2> with
	 * VecDouble4Avx2.
	 *
	 * @sa Vec
	 * @sa VecDouble4Avx2
	 */
	template<>
	struct VectorSelector<double, 4, ark::hal::simd::Avx2>
	{
		using type = VecDouble4Avx2;
	};
}


//========================================================================
#endif
