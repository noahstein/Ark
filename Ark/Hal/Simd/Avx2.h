/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel AVX2 ISA.
 * 
 * @details Intel's AVX2 spec extends the features of its first AVX spec, 
 * most notably with many additional integer functions and the fused 
 * multiply-add functions.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2,AVX,AVX2
 * @sa https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_AVX2_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_AVX2_H_INCLUDE_GUARD

#define SIMD_HAS_AVX2


//************************************************************************
//  Dependencies
//************************************************************************
#include "Avx.h"


//************************************************************************
//	Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the AVX2 ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for AVX2-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Avx2 : public Avx
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the AVX2 ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the AVX2 or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Avx2Family = AvxFamily<SIMD> && std::derived_from<SIMD, Avx2>;
}


//========================================================================
#endif
