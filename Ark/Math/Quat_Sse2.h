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


	//----------------------------------------------------------------
	//	Negation
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator-(Quat<double, ark::hal::simd::Sse2> q)
	{
		__m128d z = _mm_setzero_pd();
		__m128d wx = _mm_sub_pd(z, q.SseWx());
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Conjugate
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator*(Quat<double, ark::hal::simd::Sse2> q)
	{
		__m128d z = _mm_setzero_pd();
		__m128d wxi = q.SseWx();
		__m128d wxn = _mm_sub_pd(z, wxi);
		__m128d wx = _mm_move_sd(wxn, wxi);
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Inverse
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> Inverse(Quat<double, ark::hal::simd::Sse2> q)
	{
		return *q / Dot(q, q);
	}


	//----------------------------------------------------------------
	//	Addition
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator+(Quat<double, ark::hal::simd::Sse2> lhs, Quat<double, ark::hal::simd::Sse2> rhs)
	{
		__m128d wx = _mm_add_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_add_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Subtraction
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator-(Quat<double, ark::hal::simd::Sse2> lhs, Quat<double, ark::hal::simd::Sse2> rhs)
	{
		__m128d wx = _mm_sub_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_sub_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Quaternion-Scalar Multiplication
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator*(Quat<double, ark::hal::simd::Sse2> lhs, double rhs)
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_mul_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_mul_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Scalar-Quaternion Multiplication
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator*(double lhs, Quat<double, ark::hal::simd::Sse2> rhs)
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d wx = _mm_mul_pd(scalar, rhs.SseWx());
		__m128d yz = _mm_mul_pd(scalar, rhs.SseYz());
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Quaternion-Scalar Division
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator/(Quat<double, ark::hal::simd::Sse2> lhs, double rhs)
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_div_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_div_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Quaternion Multiplication
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator*(Quat<double, ark::hal::simd::Sse2> lhs, Quat<double, ark::hal::simd::Sse2> rhs)
	{
		__m128d z  = _mm_set1_pd(0.0);
		__m128d n  = _mm_set1_pd(-0.0);
		__m128d n0 = _mm_unpackhi_pd(n, z);
		__m128d n1 = _mm_unpackhi_pd(z, n);

		__m128d lwx = lhs.SseWx();
		__m128d lyz = lhs.SseYz();
		__m128d rwx = rhs.SseWx();
		__m128d ryz = rhs.SseYz();

		__m128d rxw = _mm_shuffle_pd(rwx, rwx, _MM_SHUFFLE2(0, 1));
		__m128d rzy = _mm_shuffle_pd(ryz, ryz, _MM_SHUFFLE2(0, 1));

		__m128d lw = _mm_unpacklo_pd(lwx, lwx);
		__m128d lx = _mm_unpackhi_pd(lwx, lwx);
		__m128d ly = _mm_unpacklo_pd(lyz, lyz);
		__m128d lz = _mm_unpackhi_pd(lyz, lyz);

		__m128d awx0   = _mm_mul_pd(lw, rwx);

		__m128d awx1r  = _mm_xor_pd(rxw, n0);
		__m128d awx1   = _mm_mul_pd(lx, awx1r);

		__m128d awx2r  = _mm_xor_pd(ryz, n0);
		__m128d awx2   = _mm_mul_pd(ly, awx2r);

		__m128d awx3r  = _mm_xor_pd(rzy, n);
		__m128d awx3   = _mm_mul_pd(lz, awx3r);

		__m128d awxa   = _mm_add_pd(awx0, awx1);
		__m128d awxb   = _mm_add_pd(awx2, awx3);
		__m128d wx     = _mm_add_pd(awxa, awxb);

		__m128d ayz0  = _mm_mul_pd(lw, ryz);

		__m128d ayz1r = _mm_xor_pd(rzy, n0);
		__m128d ayz1  = _mm_mul_pd(lx, ayz1r);

		__m128d ayz2r = _mm_xor_pd(rwx, n1);
		__m128d ayz2  = _mm_mul_pd(ly, ayz2r);

		__m128d ayz3  = _mm_mul_pd(lz, rxw);

		__m128d ayza  = _mm_add_pd(ayz0, ayz1);
		__m128d ayzb  = _mm_add_pd(ayz2, ayz3);
		__m128d yz    = _mm_add_pd(ayza, ayzb);

		return {wx, yz};
	}


	//----------------------------------------------------------------
	//	Quaternion Division
	//----------------------------------------------------------------
	inline Quat<double, ark::hal::simd::Sse2> operator/(Quat<double, ark::hal::simd::Sse2> lhs, Quat<double, ark::hal::simd::Sse2> rhs)
	{
		return lhs * Inverse(rhs);
	}
}


//========================================================================
#endif
