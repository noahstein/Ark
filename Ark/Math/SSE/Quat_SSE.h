/*========================================================================
Description
	Optimized specializations of Quat for CPUs with SSE instructions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
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
		class Quat<float>
		{
			__m128 value_;

			Quat(__m128 value)
				: value_(value)
			{}

			friend Quat operator-(Quat q);
			friend Quat operator*(Quat q);

			friend Quat operator*(Quat lhs, float rhs);
			friend Quat operator*(float lhs, Quat rhs);
			friend Quat operator+(Quat lhs, Quat rhs);
			friend Quat operator-(Quat lhs, Quat rhs);

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

			Scalar w() const { return _mm_cvtss_f32(value_); }
			Scalar x() const { return _mm_cvtss_f32(_mm_shuffle_ps(value_, value_, _MM_SHUFFLE(1, 1, 1, 1))); }
			Scalar y() const { return _mm_cvtss_f32(_mm_shuffle_ps(value_, value_, _MM_SHUFFLE(2, 2, 2, 2))); }
			Scalar z() const { return _mm_cvtss_f32(_mm_shuffle_ps(value_, value_, _MM_SHUFFLE(3, 3, 3, 3))); }
		};


		//----------------------------------------------------------------
		//	Negation
		//----------------------------------------------------------------
		inline Quat<float> operator-(Quat<float> q)
		{
			__m128 value = q.value_;
			return Quat<float>(_mm_sub_ps(_mm_xor_ps(value, value), value));
		}


		//----------------------------------------------------------------
		//	Conjugate
		//----------------------------------------------------------------
		inline Quat<float> operator*(Quat<float> q)
		{
			__m128 result = _mm_move_ss((-q).value_, q.value_);
			return result;
		}


		//----------------------------------------------------------------
		//	Addition
		//----------------------------------------------------------------
		inline Quat<float> operator+(Quat<float> lhs, Quat<float> rhs)
		{
			return Quat<float>(_mm_add_ps(lhs.value_, rhs.value_));
		}


		//----------------------------------------------------------------
		//	Subtraction
		//----------------------------------------------------------------
		inline Quat<float> operator-(Quat<float> lhs, Quat<float> rhs)
		{
			return Quat<float>(_mm_sub_ps(lhs.value_, rhs.value_));
		}


		//----------------------------------------------------------------
		//	Quaternion-Scalar Multiplication
		//----------------------------------------------------------------
		inline Quat<float> operator*(Quat<float> lhs, float rhs)
		{
			__m128 scalar = _mm_set_ps(rhs, rhs, rhs, rhs);
			return _mm_mul_ps(scalar, lhs.value_);
		}


		//----------------------------------------------------------------
		//	Scalar-Quaternion Multiplication
		//----------------------------------------------------------------
		inline Quat<float> operator*(float lhs, Quat<float> rhs)
		{
			__m128 scalar = _mm_set_ps(lhs, lhs, lhs, lhs);
			return _mm_mul_ps(scalar, rhs.value_);
		}
	}
}
