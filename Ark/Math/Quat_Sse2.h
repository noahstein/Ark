/*========================================================================
  Description
  Optimized specializations of Quat for CPUs with the SSE2 ISA. The 
  important additions for quaternions is the support of support for 
  2-element double SIMD instructions using the same 128-bit register set 
  as the original SSE ISA.

  Copyright
  Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include "Quat_Sse.h"


/*========================================================================
  Code
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
	  SSE 2 Quat<float> is structurally-identical to SSE 1
	--------------------------------------------------------------------*/
	template<> class Quat<float, ark::hal::simd::Sse2> : public Quat<float, ark::hal::simd::Sse>
	{
		using Quat<float, ark::hal::simd::Sse>::Quat;
	};


	/*--------------------------------------------------------------------
	  SSE-optimized Quat specialization for float components
	--------------------------------------------------------------------*/
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


	/*--------------------------------------------------------------------
	  Negation
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m128d z = _mm_setzero_pd();
		__m128d wx = _mm_sub_pd(z, q.SseWx());
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Conjugate
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator*(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m128d z = _mm_setzero_pd();
		__m128d wxi = q.SseWx();
		__m128d wxn = _mm_sub_pd(z, wxi);
		__m128d wx = _mm_move_sd(wxn, wxi);
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto Dot(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> double
	{
		__m128d w2x2 = _mm_mul_pd(lhs.SseWx(), rhs.SseWx());
		__m128d x2w2 = _mm_shuffle_pd(w2x2, w2x2, _MM_SHUFFLE2(0, 1));
		__m128d wx2wx2 = _mm_add_pd(w2x2, x2w2);

		__m128d y2z2 = _mm_mul_pd(lhs.SseYz(), rhs.SseYz());
		__m128d z2y2 = _mm_shuffle_pd(y2z2, y2z2, _MM_SHUFFLE2(0, 1));
		__m128d yz2yz2 = _mm_add_pd(y2z2, z2y2);

		__m128d dp = _mm_add_pd(wx2wx2, yz2yz2);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}

	/*--------------------------------------------------------------------
	  Inverse
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto Inverse(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		return *q / Dot(q, q);
	}


	/*--------------------------------------------------------------------
	Addition
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator+(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d wx = _mm_add_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_add_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Subtraction
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d wx = _mm_sub_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_sub_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator*(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_mul_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_mul_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Scalar-Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator*(double lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d wx = _mm_mul_pd(scalar, rhs.SseWx());
		__m128d yz = _mm_mul_pd(scalar, rhs.SseYz());
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Division
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_div_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_div_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	auto operator*(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		// Gather data
		__m128d n0      = _mm_set_pd(0.0, -0.0); // Negate element 0
		__m128d n1      = _mm_set_pd(-0.0, 0.0); // Negate element 1

		__m128d lwx     = lhs.SseWx();
		__m128d lyz     = lhs.SseYz();
		__m128d rwx     = rhs.SseWx();
		__m128d ryz     = rhs.SseYz();

		__m128d rxw     = _mm_shuffle_pd(rwx, rwx, _MM_SHUFFLE2(0, 1));
		__m128d rzy     = _mm_shuffle_pd(ryz, ryz, _MM_SHUFFLE2(0, 1));

		__m128d lw      = _mm_unpacklo_pd(lwx, lwx);
		__m128d lx      = _mm_unpackhi_pd(lwx, lwx);
		__m128d ly      = _mm_unpacklo_pd(lyz, lyz);
		__m128d lz      = _mm_unpackhi_pd(lyz, lyz);

		// Compute w & x components
		__m128d awx0    = _mm_mul_pd(lw, rwx);

		__m128d awx1r   = _mm_xor_pd(rxw, n0);
		__m128d awx1    = _mm_mul_pd(lx, awx1r);

		__m128d awx2r   = _mm_xor_pd(ryz, n0);
		__m128d awx2    = _mm_mul_pd(ly, awx2r);

		__m128d awx3    = _mm_mul_pd(lz, rzy);

		__m128d awx01   = _mm_add_pd(awx0, awx1);
		__m128d awx012  = _mm_add_pd(awx01, awx2);
		__m128d wx      = _mm_sub_pd(awx012, awx3);

		// Compute y & z components
		__m128d ayz0    = _mm_mul_pd(lw, ryz);

		__m128d ayz1r   = _mm_xor_pd(rzy, n0);
		__m128d ayz1    = _mm_mul_pd(lx, ayz1r);

		__m128d ayz2r   = _mm_xor_pd(rwx, n1);
		__m128d ayz2    = _mm_mul_pd(ly, ayz2r);

		__m128d ayz3    = _mm_mul_pd(lz, rxw);

		__m128d ayz01   = _mm_add_pd(ayz0, ayz1);
		__m128d ayz012  = _mm_add_pd(ayz01, ayz2);
		__m128d yz      = _mm_add_pd(ayz012, ayz3);

		return {wx, yz};
	}


	/*--------------------------------------------------------------------
	  Quaternion Division
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return lhs * Inverse(rhs);
	}
}


//========================================================================
#endif
