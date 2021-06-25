/*========================================================================
Description
	SIMD tag for the x86/x64 Intel SSE 1.0 ISA.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_HAL_SIMD_SSE_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_SSE_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include <type_traits>


/*========================================================================
  Code
========================================================================*/
namespace ark::hal::simd
{
	/*--------------------------------------------------------------------
	  SIMD architecture tag indicating he CPU has SSE2 capabilities.
	--------------------------------------------------------------------*/
	class Sse : public None
	{
	};


	/*--------------------------------------------------------------------
	  Concept to restrict templates to CPUs with the original SSE ISA.
	--------------------------------------------------------------------*/
	template<typename SIMD>
	concept IsSse = std::is_base_of_v<Sse, SIMD>;
}


//========================================================================
#endif
