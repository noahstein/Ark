/*========================================================================
Description
	Optimized specializations of Quat for CPUs with SSE2 instructions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD


//========================================================================
//	Dependencies
//========================================================================
#include "Quat_Sse.h"


//========================================================================
//	Code
//========================================================================
namespace ark
{
	namespace math
	{
		//----------------------------------------------------------------
		//	SSE2-optimized Quat specialization for float components
		//----------------------------------------------------------------
		template<>
		class Quat<float, ark::hal::Sse2> : public Quat<float, ark::hal::Sse>
		{
			friend Quat<float, ark::hal::Sse2> operator-(Quat<float, ark::hal::Sse2> q);

		protected:
			using Quat<float, ark::hal::Sse>::Quat;
		};


		//----------------------------------------------------------------
		//	Negation
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse2> operator-(Quat<float, ark::hal::Sse2> q)
		{
			std::cout << "SSE2 negation";
			__m128 value = q.Value();
			return Quat<float>(_mm_sub_ps(_mm_xor_ps(value, value), value));
		}
	}
}


//========================================================================
#endif
