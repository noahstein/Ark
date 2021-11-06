/*************************************************************************
 * @file
 * @brief SSE Vector Specializations
 * 
 * @details This file defines specializations to the Vec class for the 
 * original SSE ISA. Due to the limitations of the original register and 
 * instructions sets, this is limited to single-precision floating-point 
 * implementations of 4-dimensional vectors.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include <ranges>

#include "Vector.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*********************************************************************
	 * @brief General N-dimensional Vector
	 * 
	 * @details This class defines a simple and general template 
	 * implementation of the Vector concept. It is a dense array of data. 
	 * For architectures with SIMD ISAs, it is fully expected for 
	 * specializations to implement ISA-specific optimizations.
	 ********************************************************************/
	template<typename S, std::size_t N, typename I = ark::hal::simd::HAL_SIMD>
	class Vec
	{
	public:
		using Scalar = S;

	private:
		Scalar data_[N];

	protected:
		static constexpr auto Range()
		{
			return std::views::iota(std::size_t{ 0 }, N);
		}

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
			requires std::convertible_to<typename V::Scalar, Scalar>
			&& SameDimension<Vec, V>
		constexpr Vec(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(rhs(i)); });
		}

		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar>
				&& SameDimension<Vec, V>
		constexpr Vec& operator=(const Vec& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(rhs(i)); });
			return *this;
		}

		constexpr Vec& operator=(int value)
		{
			static_assert(value == 0, "0 is the only valid scalar value that may be used to construct a vector.");
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(value); });
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


//========================================================================
#endif
