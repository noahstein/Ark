/*========================================================================
Description
	Basic library definitions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
========================================================================*/

#pragma once


//========================================================================
//	Platform-specific Inclusion
//========================================================================
#define INCLUDE_STRINGIZE(S) #S
#define INCLUDE_BUILD_FILENAME(T, F) INCLUDE_STRINGIZE(T/F ## _ ## T.h)
#define INCLUDE_SPECIFIC(T, F) INCLUDE_BUILD_FILENAME(T, F)
#define INCLUDE_SIMD(F) INCLUDE_SPECIFIC(SIMD, F)
