/** **********************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel SSE3 ISA.
 * 
 * @details SSE3 extends the first two SSE sets with no changes to the 
 * registers or their interpretations. It adds some additional operations 
 * on the data types already present, most notably addsub variants and 
 * some move instructions.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE,SSE2SSE3
 * @sa https://en.wikipedia.org/wiki/SSE3
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_SSE3_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_SSE3_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Sse2.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/** ******************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the SSE3 ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for SSE3-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Sse3 : public Sse2
	{
	};


	/** ******************************************************************
	 * @brief Concept restricting templates to CPUs with the SSE3 ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the SSE3 or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept IsSse3 = IsSse2<SIMD> && std::is_base_of_v<Sse3, SIMD>;
}


//************************************************************************
#endif
