/*========================================================================
Description
	Hardware Abstraction Layer

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_HAL_H_INCLUDE_GUARD)
#define ARK_HAL_H_INCLUDE_GUARD


/*========================================================================
Dependencies
========================================================================*/
#include "Ark/Hal/Hal.h"


/*========================================================================
Platform-specific Inclusion
========================================================================*/

// Add quatoation marks around a string
#define INCLUDE_STRINGIZE(String) #String

// Build a platform-specific header file name
#define INCLUDE_BUILD_FILENAME(Type, File) INCLUDE_STRINGIZE(File ## _ ## Type.h)

// Include a platform-specific hardware abstraction layer header file
#define INCLUDE_HAL(Type, File) INCLUDE_BUILD_FILENAME(Type, File)

// Include a platform-specific SIMD header file F
#define INCLUDE_SIMD(File) INCLUDE_HAL(HAL_SIMD, File)


//========================================================================
#endif
