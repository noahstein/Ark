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
namespace ark::math
{
	/*----------------------------------------------------------------
		SSE-optimized Quat specialization for float components
	----------------------------------------------------------------*/
	template<>
	class Quat<double, ark::hal::simd::Sse2>
	{
		__m128d wx_;
		__m128d yz_;

	public:
		using Scalar = float;

		Quat() = default;

		Quat(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			wx_ = _mm_set_pd(x, w);
			yz_ = _mm_set_pd(z, y);
		}

		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}

		Quat(__m128d wx, __m128d yz)
			: wx_(wx), yz_(yz)
		{}

		__m128d SseWx() const { return wx_; }
		__m128d SseYz() const { return yz_; }

		Scalar w() const { return _mm_cvtsd_f64 (SseWx()); }
		Scalar x() const { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseWx(), SseWx())); }
		Scalar y() const { return _mm_cvtsd_f64 (SseYz()); }
		Scalar z() const { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseYz(), SseYz())); }
	};


#if 0
	//----------------------------------------------------------------
	//	Negation
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator-(Quat<float, ark::hal::simd::Sse> q)
	{
		std::cerr << "SSE negation";
		__m128 value = q.SseVal();
		return Quat<float, ark::hal::simd::Sse>(_mm_sub_ps(_mm_xor_ps(value, value), value));
	}


	//----------------------------------------------------------------
	//	Conjugate
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator*(Quat<float, ark::hal::simd::Sse> q)
	{
		__m128 result = _mm_move_ss((-q).SseVal(), q.SseVal());
		return result;
	}


	//----------------------------------------------------------------
	//	Inverse
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> Inverse(Quat<float, ark::hal::simd::Sse> q)
	{
		return *q / Dot(q, q);
	}


	//----------------------------------------------------------------
	//	Addition
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator+(Quat<float, ark::hal::simd::Sse> lhs, Quat<float, ark::hal::simd::Sse> rhs)
	{
		return Quat<float, ark::hal::simd::Sse>(_mm_add_ps(lhs.SseVal(), rhs.SseVal()));
	}


	//----------------------------------------------------------------
	//	Subtraction
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator-(Quat<float, ark::hal::simd::Sse> lhs, Quat<float, ark::hal::simd::Sse> rhs)
	{
		return Quat<float, ark::hal::simd::Sse>(_mm_sub_ps(lhs.SseVal(), rhs.SseVal()));
	}


	//----------------------------------------------------------------
	//	Quaternion-Scalar Multiplication
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator*(Quat<float, ark::hal::simd::Sse> lhs, float rhs)
	{
		__m128 scalar = _mm_set1_ps(rhs);
		return _mm_mul_ps(scalar, lhs.SseVal());
	}


	//----------------------------------------------------------------
	//	Scalar-Quaternion Multiplication
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator*(float lhs, Quat<float, ark::hal::simd::Sse> rhs)
	{
		__m128 scalar = _mm_set1_ps(lhs);
		return _mm_mul_ps(scalar, rhs.SseVal());
	}


	//----------------------------------------------------------------
	//	Quaternion-Scalar Division
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator/(Quat<float, ark::hal::simd::Sse> lhs, float rhs)
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_div_ps(lhs.SseVal(), scalar);
		return result;
	}


	//----------------------------------------------------------------
	//	Quaternion Multiplication
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator*(Quat<float, ark::hal::simd::Sse> lhs, Quat<float, ark::hal::simd::Sse> rhs)
	{
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();
		__m128 n = _mm_set1_ps(-0.0);
		__m128 z = _mm_setzero_ps();
		__m128 s = _mm_shuffle_ps(z, n, _MM_SHUFFLE(0, 0, 0, 0));

		__m128 l_w = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 a_w = _mm_mul_ps(l_w, r);

		__m128 l_x = _mm_shuffle_ps(l, l, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 r_b = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1));
		__m128 r_j = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 2, 0, 2));
		__m128 r_t = _mm_xor_ps(r_b, r_j);
		__m128 a_x = _mm_mul_ps(l_x, r_t);

		__m128 l_y = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 r_c = _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 0, 3, 2));
		__m128 r_k = _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 0, 0, 2));
		__m128 r_u = _mm_xor_ps(r_c, r_k);
		__m128 a_y = _mm_mul_ps(l_y, r_u);

		__m128 l_z = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 r_d = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 1, 2, 3));
		__m128 r_l = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 2, 2));
		__m128 r_v = _mm_xor_ps(r_d, r_l);
		__m128 a_z = _mm_mul_ps(l_z, r_v);

		__m128 a_1 = _mm_add_ps(a_w, a_x);
		__m128 a_2 = _mm_add_ps(a_y, a_z);
		__m128 a = _mm_add_ps(a_1, a_2);

		return a;
	}


	//----------------------------------------------------------------
	//	Quaternion Division
	//----------------------------------------------------------------
	inline Quat<float, ark::hal::simd::Sse> operator/(Quat<float, ark::hal::simd::Sse> lhs, Quat<float, ark::hal::simd::Sse> rhs)
	{
		return lhs * Inverse(rhs);
	}

#endif
}


//========================================================================
#endif
