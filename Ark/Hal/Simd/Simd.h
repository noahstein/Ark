/*========================================================================
Description
	SIMD ISA tag.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#pragma once

//========================================================================
//	Dependencies
//========================================================================
#include "../Hal.h"


//========================================================================
//	Code
//========================================================================
namespace ark
{
	namespace hal
	{
		class Simd
		{
		};

		// TODO: Remove when include worked out
		class Sse : public Simd
		{
		};

		class Sse2 : public Sse
		{
		};

		class Sse3 : public Sse2
		{
		};

		class Ssse3 : public Sse3
		{
		};

		class Sse4_1 : public Ssse3
		{
		};

		class Sse4_2 : public Sse4_1
		{
		};
	}
}
