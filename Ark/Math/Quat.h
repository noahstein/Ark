/*========================================================================
Description
	Straightforward implementation of the Quaternion concept. This 
	implementation will include platform-specific specializations 
	defined in other header files.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
========================================================================*/

#pragma once


/*========================================================================
	Dependencies
========================================================================*/
#include "Quaternion.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark
{
	namespace math
	{
		//----------------------------------------------------------------
		// Simple 4-component quaternion implementation
		//----------------------------------------------------------------
		template<typename S, typename I = ark::hal::HAL_SIMD>
		class Quat
		{
		public:
			using Scalar = S;

		private:
			Scalar w_, x_, y_, z_;

		public:
			Quat() noexcept
			{}

			Quat(Scalar w, Scalar x, Scalar y, Scalar z) noexcept(std::is_nothrow_copy_constructible_v<Scalar>)
				: w_(w), x_(x), y_(y), z_(z)
			{}

			template<Quaternion Q>
				requires std::convertible_to<typename Q::Scalar, Scalar>
			Quat(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
				: Quat(Scalar{rhs.w()}, Scalar{rhs.x()}, Scalar{rhs.y()}, Scalar{rhs.z()})
			{}

			template<Quaternion Q>
				requires std::convertible_to<typename Q::Scalar, Scalar>
			Quat& operator=(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
			{
				w_ = Scalar{rhs.w()};
				x_ = Scalar{rhs.x()};
				y_ = Scalar{rhs.y()};
				z_ = Scalar{rhs.z()};

				return *this;
			}

			Scalar w() const { return w_; }
			Scalar x() const { return x_; }
			Scalar y() const { return y_; }
			Scalar z() const { return z_; }
		};
	}
}

//========================================================================
//	Platform-optimized Specializations
//========================================================================
#if __has_include(INCLUDE_SIMD(Quat))
#include INCLUDE_SIMD(Quat)
#endif