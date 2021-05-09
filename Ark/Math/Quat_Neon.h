// Currently a place-holder for implementing multi-platform features

#pragma once

namespace ark
{
	namespace math
	{
		template<>
		class Quat<float>
		{
			float value_[4];

			Quat(float value[4])
			{
				value_[0] = value[0];
				value_[1] = value[1];
				value_[2] = value[2];
				value_[3] = value[3];
			}

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
				value_[0] = ww;
				value_[1] = xx;
				value_[2] = yy;
				value_[3] = zz;
			}

			template<Quaternion Q>
			Quat(const Q& rhs)
				: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
			{}

			Scalar w() const { return value_[0]; }
			Scalar x() const { return value_[1]; }
			Scalar y() const { return value_[2]; }
			Scalar z() const { return value_[3]; }
		};

		inline Quat<float> operator-(Quat<float> q)
		{
			const float* v = q.value_;
			return Quat(v[0], v[1], v[2], v[3]);
		}
		
#if 0
		inline Quat<float> operator*(Quat<float> q)
		{
			__m128 value = q.value_;
			__m128 neg = _mm_sub_ps(_mm_xor_ps(value, value), value);
			__m128 result = _mm_move_ss(neg, value);
			return result;
		}

		inline Quat<float> operator*(Quat<float> lhs, float rhs)
		{
			__m128 scalar = _mm_set_ps(rhs, rhs, rhs, rhs);
			return _mm_mul_ps(scalar, lhs.value_);
		}

		inline Quat<float> operator*(float lhs, Quat<float> rhs)
		{
			__m128 scalar = _mm_set_ps(lhs, lhs, lhs, lhs);
			return _mm_mul_ps(scalar, rhs.value_);
		}

		inline Quat<float> operator+(Quat<float> lhs, Quat<float> rhs)
		{
			return Quat<float>(_mm_add_ps(lhs.value_, rhs.value_));
		}

		inline Quat<float> operator-(Quat<float> lhs, Quat<float> rhs)
		{
			return Quat<float>(_mm_sub_ps(lhs.value_, rhs.value_));
		}
#endif		
	}
}
