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
	public:
		using Scalar = S;

	private:
		Scalar data_[N];

	public:
		constexpr Vec() noexcept(std::is_nothrow_constructible_v<Scalar>) = default;

		template<typename ... T>
			requires (sizeof...(T) == N)
		constexpr Vec(T && ... values) noexcept((std::is_nothrow_convertible_v<T, Scalar> && ...))
		{
			auto Set = [this, i = 0] <typename T> (T && value) mutable
			{
				data_[i++] = static_cast<Scalar>(std::forward<T>(value));
			};

			(Set(values), ...);
		}

		template<Vector V>
			requires SameDimension<Vec, V>	
		constexpr Vec(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			for (std::size_t i = 0; i < Size(); ++i)
			{
				data_[i] = static_cast<Scalar>(rhs(i));
			}
		}

		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && (V::N == N)
		constexpr Vec& operator=(const Vec& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			for (std::size_t i = 0; i < Size(); ++i)
			{
				data_[i] = static_cast<Scalar>(rhs(i));
			}
			return *this;
		}

		constexpr Vec& operator=(const Scalar value)
		{
			static_assert(value == 0, "0 is the only valid scalar value that may be used to construct a vector.");
			for (std::size_t i = 0; i < Size(); ++i)
			{
				data_[i] = value;
			}
		}

		static constexpr size_t Size() noexcept { return N; }
		constexpr Scalar operator()(size_t index) const noexcept(std::is_nothrow_copy_constructible_v<Scalar>)
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