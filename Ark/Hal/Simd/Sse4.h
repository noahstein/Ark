/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel SSE4 ISA.
 * 
 * @details SSE4, like SSE3, does not extend the register set nor their 
 * interpretations. Intel defined two upgrades to the SSE archecture in 
 * the 4th generation. This includes both SSE 4.1 and 4.2.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2SSE3,SSE4_1,SSE4_2
 * @sa https://en.wikipedia.org/wiki/SSE4
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_SSE4_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_SSE4_H_INCLUDE_GUARD

#define SIMD_HAS_SSE4


//************************************************************************
//  Dependencies
//************************************************************************
#include "Sse3.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the SSE4.2 ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for SSE4-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Sse4 : public Sse3
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the SSE4.2 ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the SSE4 or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Sse4Family = Sse3Family<SIMD> && std::is_base_of_v<Sse4, SIMD>;
}


//************************************************************************
#endif
