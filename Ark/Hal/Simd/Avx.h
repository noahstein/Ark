/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel AVX ISA.
 * 
 * @details Intel's AVX spec for SIMD operations is not a brand new 
 * instruction set. It is a superset of SSE 4.1. It extends the registers 
 * to 256-bits and doubles the number of them to 16. It also upgrades 
 * functions to use 3 operands.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2,AVX
 * @sa https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_AVX_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_AVX_H_INCLUDE_GUARD

#define SIMD_HAS_AVX


//************************************************************************
//  Dependencies
//************************************************************************
#include "Sse4.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the AVX ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for AVX-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Avx : public Sse4
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the AVX ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the AVX or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept AvxFamily = Sse4Family<SIMD> && std::is_base_of_v<Avx, SIMD>;
}


//========================================================================
#endif
