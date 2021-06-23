/*========================================================================
Description
	Straightforward implementation of the Matrix concept. This 
	implementation will include platform-specific specializations 
	defined in other header files.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_MTX_H_INCLUDE_GUARD)
#define ARK_MATH_MTX_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include "Matrix.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
	 Simple, Dense Statically-sized Matrix
	--------------------------------------------------------------------*/
	template<typename S, int R, int C, typename I = ark::hal::HAL_SIMD>
	class Mtx
	{
	public:
		using Scalar = S;

	private:
		S data_[R][C];

	public:
		constexpr Mtx()
		{}

		template<typename ... T>
			requires (sizeof...(T) == R * C)
		constexpr Mtx(T && ... values)
		{
			auto Set = [this, r = 0, c = 0] <typename T> (T && value) mutable
			{
				data_[r][c] = static_cast<Scalar>(std::forward<T>(value));
				if (++c == Width())
				{
					c = 0;
					++r;
				}
			};

			(Set(values), ...);
		}

		template<Matrix M>
			requires std::convertible_to<typename M::Scalar, Scalar>
			&& (M::N == N)
		constexpr Mtx& operator=(const Mtx& rhs)
		{
			for (std::size_t r = 0; r < rhs.Height(); ++r)
			{
				for (std::size_t c = 0; c < rhs.Width(); ++c)
				{
					data_[r][c] = rhs(r, c);					
				}
			}
			return *this;
		}

		static constexpr std::size_t Width() { return C; }
		static constexpr std::size_t Height() { return R; }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		 {
			  return data_[row][column];
		}
	};
}

/*========================================================================
	Platform-optimized Specializations
========================================================================*/
#if __has_include(INCLUDE_SIMD(Vec))
#include INCLUDE_SIMD(Vec)
#endif


//========================================================================
#endif
