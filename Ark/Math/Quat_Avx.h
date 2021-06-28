/*========================================================================
Description
  Optimized specializations of Quat for CPUs with AVX instructions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_AVX_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_AVX_H_INCLUDE_GUARD


/*========================================================================
  Dependencies
========================================================================*/
#include "Quat_Sse4.h"


/*========================================================================
  Code
========================================================================*/
namespace ark::math
{
	/*====================================================================
	  Classes
	====================================================================*/

	/*--------------------------------------------------------------------
	  AVX Quat<float> is structurally-identical to that of SSE
	--------------------------------------------------------------------*/
	template<> class Quat<float, ark::hal::simd::Avx> : public Quat<float, ark::hal::simd::Sse4>
	{
		using Quat<float, ark::hal::simd::Sse4>::Quat;
	};


	/*--------------------------------------------------------------------
	  AVX Quat<double> takes advantage of new 256-bit AVX registers,
	  wide enough to hold all 4 doubles.
	--------------------------------------------------------------------*/
	template<> class Quat<double, ark::hal::simd::Avx>
	{
		__m256d value_;

	public:
		using Scalar = double;

		Quat() = default;

		Quat(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			value_ = _mm256_set_pd(z, y, x, w);
		}

		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}

		Quat(__m256d value)
			: value_(value)
		{}

		__m256d AvxVal() const { return value_; }

		Scalar w() const
		{
			return _mm256_cvtsd_f64(AvxVal());
		}
	
		Scalar x() const
		{
			__m256d x = _mm256_permute_pd(AvxVal(), 1);
			return _mm256_cvtsd_f64(x);
		}
	
		Scalar y() const
		{
			__m256d val = AvxVal();
			__m256d y = _mm256_permute2f128_pd(val, val, 1);
			double result = _mm256_cvtsd_f64(y);
			return result;
		}
	
		Scalar z() const
		{
			__m256d val = AvxVal();
			__m256d yz = _mm256_permute2f128_pd(val, val, 1);
			__m256d y = _mm256_permute_pd(yz, 1);
			double result = _mm256_cvtsd_f64(y);
			return result;
		}
	};


	/*====================================================================
	  Functions
	====================================================================*/

	/*--------------------------------------------------------------------
	  Negation
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m256d zero = _mm256_setzero_pd();
		__m256d result = _mm256_sub_pd(zero, q.AvxVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Conjugate
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m256d val = q.AvxVal();
		__m256d neg = (-q).AvxVal();
		__m256d result = _mm256_blend_pd(val, neg, 0b1110);
		return result;
	}


	/*--------------------------------------------------------------------
		Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto Dot(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> double
	{
		__m256d w_x_y_z = _mm256_mul_pd(lhs.AvxVal(), rhs.AvxVal());
		__m256d wx_yz = _mm256_hadd_pd(w_x_y_z, w_x_y_z);
		__m256d yz_wx = _mm256_permute2f128_pd(wx_yz, wx_yz, 5);
		__m256d wxyz = _mm256_add_pd(wx_yz, yz_wx);
		return _mm256_cvtsd_f64(wxyz);
	}


	/*--------------------------------------------------------------------
	  Addition
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator+(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return _mm256_add_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*--------------------------------------------------------------------
	  Subtraction
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return _mm256_sub_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_mul_pd(scalar, lhs.AvxVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Scalar-Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(double lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(lhs);
		__m256d result = _mm256_mul_pd(scalar, rhs.AvxVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Division
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_div_pd(lhs.AvxVal(), scalar);
		return result;
	}


	/*--------------------------------------------------------------------
		Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsAvx SIMD>
	auto operator*(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		// Gather data
		__m256d l      = lhs.AvxVal();
		__m256d r      = rhs.AvxVal();

		// Might just be faster to broadcast from memory

		__m256d lw_lx  = _mm256_permute2f128_pd(l , l, 0);    // lw, lx, lw, lx
		__m256d lw     = _mm256_permute_pd(lw_lx, 0);         // lw, lw, lw, lw
		__m256d lx     = _mm256_permute_pd(lw_lx, 0xF);       // lx, lx, lx, lx

		__m256d ly_lz  = _mm256_permute2f128_pd (l , l, 17);  // ly, lz, ly, lz
		__m256d ly     = _mm256_permute_pd(ly_lz, 0);         // ly, ly, ly, ly
		__m256d lz     = _mm256_permute_pd(ly_lz, 0xF);       // lz, lz, lz, lz

		// Compute partial sum first column
		__m256d ps0    = _mm256_mul_pd(lw, r);                // lw*rw, lx*rx, ly*ry, lz*rz

		// Compute partial sum second column
		__m256d r_xwzy = _mm256_permute_pd(r, 5);             // rx, rw, rz, ry
		__m256d ps1    = _mm256_mul_pd(lx, r_xwzy);           // lx*rx, lx*rw, lx*rz, lx*ry

		// Compute partial sum third column
		__m256d r_yzwx = _mm256_permute2f128_pd(r, r, 1);     // ry, rz, rw, rx
		__m256d n2     = _mm256_set_pd(-0.0, 0.0, 0.0, -0.0); // -, +, +, -
		__m256d r_2n   = _mm256_xor_pd(r_yzwx, n2);           // -ry, rz, rw, -rx
		__m256d ps2    = _mm256_mul_pd(ly, r_2n);             // -ly*ry, ly*rz, ly*rw, -ly*rx

		// Compute partial sum fourth column
		__m256d r_zyxw = _mm256_permute_pd(r_yzwx, 5);        // rz, ry, rx, rw
		__m256d n3     = _mm256_permute_pd(n2, 0);            // -, -, +, +
		__m256d r_3n   = _mm256_xor_pd(r_zyxw, n3);           // -rx, -ry, rx, rw
		__m256d ps3    = _mm256_mul_pd(lz, r_3n);             // -lz*rx, -lz*ry, lz*rx, lz*rw

		// Combine column partial sums into result
		__m256d ps01   = _mm256_addsub_pd(ps0, ps1);
		__m256d ps012  = _mm256_add_pd(ps01, ps2);
		__m256d a      = _mm256_add_pd(ps012, ps3);

		return a;
	}
}


//========================================================================
#endif
