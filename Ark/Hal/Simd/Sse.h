/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the x86/x64 Intel SSE (1.0) ISA.
 * 
 * @details SSE expanded the x86 ISA with floating-point SIMD 
 * functionality as well as some extra MMX functions. It added 8 128-bit 
 * registers, divided into 4 32-bit single-precision floating-point 
 * values. The tag and concept defined in this header are used to mark 
 * template classes and functions to be used when compiling for targets 
 * with SSE.
 * 
 * @sa @ref SimdArchitecture
 * @sa https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE
 * @sa https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions\
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_SSE_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_SSE_H_INCLUDE_GUARD

#define SIMD_HAS_SSE


//************************************************************************
// Dependencies
//************************************************************************
#include <type_traits>


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the SSE ISA.
	 * 
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for SSE-specific optimizations.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Sse : public None
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the SSEs ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for AVX-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the SSE or other 
	 * compatible SIMD ISA.
	 * 
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept SseFamily = std::derived_from<SIMD, Sse>;
}


//************************************************************************
#endif
