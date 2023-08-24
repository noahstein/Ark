/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel SSE2 ISA.
 * 
 * @details SSE2 expands the functions of the original SSE. It adds no 
 * new registers; however, it adds new single-precision functions as well 
 * as 2-element double-precision functions and integer functions for the 
 * full 128-bit registers to supplant the older MMX ones.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2
 * @sa https://en.wikipedia.org/wiki/SSE2
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_SSE2_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_SSE2_H_INCLUDE_GUARD

#define SIMD_HAS_SSE2


//************************************************************************
// Dependencies
//************************************************************************
#include "Sse.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the SSE2 ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for SSE2-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Sse2 : public Sse
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the SSE2 ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the SSE2 or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Sse2Family = SseFamily<SIMD> && std::derived_from<SIMD, Sse2>;
}


//************************************************************************
#endif
