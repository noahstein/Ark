/*========================================================================
Description
	SIMD base ISA tag. Specific SIMD architecture tags will derive from 
	this tag. These tags will get defined in architecture-specific files, 
	not mererly one per architecture, but one per architecture revision 
	to ensure maximum encapsulation.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_HAL_SIMD_SIMD_H)
#define ARK_HAL_SIMD_SIMD_H


//========================================================================
//	Dependencies
//========================================================================
#include "../Hal.h"


//========================================================================
//	Code
//========================================================================
namespace ark::hal::simd
{
	class None
	{
	};
}



/*========================================================================
	Platform-optimized Specializations
========================================================================*/
#if __has_include(INCLUDE_HAL_LOCAL(HAL_SIMD))
#include INCLUDE_HAL_LOCAL(HAL_SIMD)
#endif

//========================================================================
#endif
