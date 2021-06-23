/*========================================================================
Description
	Optimized specializations of Quat for CPUs with SSE instructions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#pragma once


//========================================================================
//	Dependencies
//========================================================================
#include <immintrin.h>


//========================================================================
//	Code
//========================================================================
namespace ark
{
	namespace math
	{
		//----------------------------------------------------------------
		//	SSE-optimized Quat specialization for float components
		//----------------------------------------------------------------
		template<>
		class Quat<float, ark::hal::Sse>
		{
			__m128 value_;

			friend Quat<float, ark::hal::Sse> operator-(Quat<float, ark::hal::Sse> q);
			friend Quat operator*(Quat q);

			friend Quat operator*(Quat lhs, float rhs);
			friend Quat operator*(float lhs, Quat rhs);
			friend Quat operator+(Quat lhs, Quat rhs);
			friend Quat operator-(Quat lhs, Quat rhs);

		protected:
			Quat(__m128 value)
				: value_(value)
			{}

			__m128 Value() const { return value_; }

		public:
			using Scalar = float;

			Quat()
			{}

			Quat(Scalar ww, Scalar xx, Scalar yy, Scalar zz)
			{
				value_ = _mm_setr_ps(ww, xx, yy, zz);
			}

			template<Quaternion Q>
			Quat(const Q& rhs)
				: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
			{}

			Scalar w() const { return _mm_cvtss_f32(Value()); }
			Scalar x() const { return _mm_cvtss_f32(_mm_shuffle_ps(Value(), Value(), _MM_SHUFFLE(1, 1, 1, 1))); }
			Scalar y() const { return _mm_cvtss_f32(_mm_shuffle_ps(Value(), Value(), _MM_SHUFFLE(2, 2, 2, 2))); }
			Scalar z() const { return _mm_cvtss_f32(_mm_shuffle_ps(Value(), Value(), _MM_SHUFFLE(3, 3, 3, 3))); }
		};


		//----------------------------------------------------------------
		//	Negation
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator-(Quat<float, ark::hal::Sse> q)
		{
			std::cerr << "SSE negation";
			__m128 value = q.Value();
			return Quat<float, ark::hal::Sse>(_mm_sub_ps(_mm_xor_ps(value, value), value));
		}


		//----------------------------------------------------------------
		//	Conjugate
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator*(Quat<float, ark::hal::Sse> q)
		{
			__m128 result = _mm_move_ss((-q).Value(), q.Value());
			return result;
		}


		//----------------------------------------------------------------
		//	Addition
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator+(Quat<float, ark::hal::Sse> lhs, Quat<float, ark::hal::Sse> rhs)
		{
			return Quat<float, ark::hal::Sse>(_mm_add_ps(lhs.Value(), rhs.Value()));
		}


		//----------------------------------------------------------------
		//	Subtraction
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator-(Quat<float, ark::hal::Sse> lhs, Quat<float, ark::hal::Sse> rhs)
		{
			return Quat<float, ark::hal::Sse>(_mm_sub_ps(lhs.Value(), rhs.Value()));
		}


		//----------------------------------------------------------------
		//	Quaternion-Scalar Multiplication
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator*(Quat<float, ark::hal::Sse> lhs, float rhs)
		{
			__m128 scalar = _mm_set_ps(rhs, rhs, rhs, rhs);
			return _mm_mul_ps(scalar, lhs.Value());
		}


		//----------------------------------------------------------------
		//	Scalar-Quaternion Multiplication
		//----------------------------------------------------------------
		inline Quat<float, ark::hal::Sse> operator*(float lhs, Quat<float, ark::hal::Sse> rhs)
		{
			__m128 scalar = _mm_set_ps(lhs, lhs, lhs, lhs);
			return _mm_mul_ps(scalar, rhs.Value());
		}
	}
}
