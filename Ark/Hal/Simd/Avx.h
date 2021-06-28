/*========================================================================
Description
	SIMD tag for the x86/x64 Intel AVX ISA.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_HAL_SIMD_AVX_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_AVX_H_INCLUDE_GUARD


//========================================================================
//	Dependencies
//========================================================================
#include "Sse4.h"


//========================================================================
//	Code
//========================================================================
namespace ark::hal::simd
{
	/*--------------------------------------------------------------------
	  SIMD architecture tag indicating he CPU has AVX capabilities.
	--------------------------------------------------------------------*/
	class Avx : public Sse4
	{
	};


	/*--------------------------------------------------------------------
	  Concept to restrict templates to CPUs with the AVX ISA.
	--------------------------------------------------------------------*/
	template<typename SIMD>
	concept IsAvx = IsSse4<SIMD> && std::is_base_of_v<Avx, SIMD>;
}


//========================================================================
#endif
