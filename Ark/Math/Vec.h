/*========================================================================
Description
	Straightforward implementation of the Vector concept. This 
	implementation will include platform-specific specializations 
	defined in other header files.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
========================================================================*/

#pragma once


/*========================================================================
	Dependencies
========================================================================*/
#include "Vector.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
	 Simple, Dense Statically-sized Vector
	--------------------------------------------------------------------*/
	template<typename S, int N, typename I = ark::hal::HAL_SIMD>
	class Vec
	{
		S data_[N];

	public:
		using Scalar = S;

		Vec()
		{}

		constexpr Vec(const S(&values)[N])
		{
			size_t i = 0;
			for (auto v : values)
			{
				data_[i++] = v;
			}
		}

		template<Vector V>
		Vec(const V& rhs)
		{}

		template<Vector V>
		requires std::convertible_to<typename V::Scalar, Scalar> && (V::N == N)
		Vec& operator=(const Vec& rhs)
		{

			return *this;
		}

		static constexpr size_t Size() { return N; }
		constexpr S operator()(size_t index) const
		{
			return data_[index];
		}
	};
}

/*========================================================================
	Platform-optimized Specializations
========================================================================*/
#if __has_include(INCLUDE_SIMD(Vec))
#include INCLUDE_SIMD(Vec)
#endif