/*========================================================================
Description
	Straightforward implementation of the Quaternion concept. This 
	implementation will include platform-specific specializations 
	defined in other header files.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
========================================================================*/

#pragma once


//========================================================================
//	Dependencies
//========================================================================
#include "Quaternion.h"
#include "Ark/Hal/Simd/Simd.h"


//========================================================================
//	Code
//========================================================================
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
			S w_, x_, y_, z_;

		public:
			using Scalar = S;

			Quat()
			{}

			Quat(Scalar ww, Scalar xx, Scalar yy, Scalar zz)
				: w_(ww), x_(xx), y_(yy), z_(zz)
			{}

			template<Quaternion Q>
			Quat(const Q& rhs)
				: Quat(static_cast<S>(rhs.w()), static_cast<S>(rhs.x()), static_cast<S>(rhs.y()), static_cast<S>(rhs.z()))
			{}

			template<Quaternion Q>
			// Put in convertible test from Q::S -> S
			Quat& operator=(const Q& rhs)
			{
				w_ = Scalar(rhs.w());
				x_ = Scalar(rhs.x());
				y_ = Scalar(rhs.y());
				z_ = Scalar(rhs.z());

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