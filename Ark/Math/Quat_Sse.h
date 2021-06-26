/*========================================================================
Description
  Optimized specializations of Quat for CPUs with the first-generation 
  SSE ISA. For quaternions, the important ones are the SIMD data type 
  with 4 float components and associated instrucxtions. The ISA does not 
  support doubles.

Copyright
  Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUAT_SSE_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE_H_INCLUDE_GUARD


//========================================================================
//	Dependencies
//========================================================================
#include <immintrin.h>


//========================================================================
//	Code
//========================================================================
namespace ark::math
{
	/*--------------------------------------------------------------------
	  SSE-optimized Quat specialization for float components
	--------------------------------------------------------------------*/
	template<>
	class Quat<float, ark::hal::simd::Sse>
	{
		__m128 value_;

	public:
		using Scalar = float;

		Quat() = default;

		Quat(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			value_ = _mm_setr_ps(w, x, y, z);
		}

		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}

		Quat(__m128 value)
			: value_(value)
		{}

		__m128 SseVal() const { return value_; }

		Scalar w() const { return _mm_cvtss_f32(SseVal()); }
		Scalar x() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(1, 1, 1, 1))); }
		Scalar y() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(2, 2, 2, 2))); }
		Scalar z() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(3, 3, 3, 3))); }
	};


	/*--------------------------------------------------------------------
	  Negation
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		__m128 result = _mm_sub_ps(_mm_setzero_ps(), q.SseVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Conjugate
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		__m128 result = _mm_move_ss((-q).SseVal(), q.SseVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Dot
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto Dot(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> float
	{
		__m128 squares = _mm_mul_ps(lhs.SseVal(), rhs.SseVal());
		__m128 badc = _mm_shuffle_ps(squares, squares, _MM_SHUFFLE(2, 3, 0, 1));
		__m128 pairs = _mm_add_ps(squares, badc);
		__m128 bbaa = _mm_shuffle_ps(pairs, pairs, _MM_SHUFFLE(0, 1, 2, 3));
		__m128 dp = _mm_add_ps(pairs, bbaa);
		float result = _mm_cvtss_f32(dp);
		return result;
	}

	/*--------------------------------------------------------------------
	  Inverse
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto Inverse(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		return *q / Dot(q, q);
	}


	/*--------------------------------------------------------------------
	  Addition
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator+(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return _mm_add_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*--------------------------------------------------------------------
	  Subtraction
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return _mm_sub_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(Quat<float, SIMD> lhs, float rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_mul_ps(scalar, lhs.SseVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Scalar-Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(float lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(lhs);
		__m128 result = _mm_mul_ps(scalar, rhs.SseVal());
		return result;
	}


	/*--------------------------------------------------------------------
	  Quaternion-Scalar Division
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator/(Quat<float, SIMD> lhs, float rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_div_ps(lhs.SseVal(), scalar);
		return result;
	}


	/*--------------------------------------------------------------------
	  Quaternion Multiplication
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	auto operator*(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		// Gather data
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();

		// Compute partial result first column
		__m128 l_w = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 0, 0, 0)); // lw, lx, ly, lz
		__m128 a_w = _mm_mul_ps(l_w, r); // lw*rw, lw*rx, lw*ry, lw*rz

		// Compute partial result second column
		__m128 l_x = _mm_shuffle_ps(l, l, _MM_SHUFFLE(1, 1, 1, 1)); // lx, lx, lx, lx
		__m128 r_b = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)); // rx, rw, rz, ry
		__m128 r_j = _mm_set_ps(0.0, -0.0, 0.0, -0.0); // -, +, -, +
		__m128 r_t = _mm_xor_ps(r_b, r_j); // -rx, rw, -rz, ry
		__m128 a_x = _mm_mul_ps(l_x, r_t); // -lx*rx, lx*rw, -lx*rz, lx*ry

		// Compute partial result third column
		__m128 l_y = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 2, 2, 2)); // ly, ly, ly, ly
		__m128 r_c = _mm_shuffle_ps(r_b, r_b, _MM_SHUFFLE(0, 1, 2, 3)); // ry, rz, rw, rx
		__m128 r_k = _mm_shuffle_ps(r_j, r_j, _MM_SHUFFLE(0, 1, 1, 0)); // -, +, +, -
		__m128 r_u = _mm_xor_ps(r_c, r_k); // -ry, rz, rw, -rx
		__m128 a_y = _mm_mul_ps(l_y, r_u); // -ly*ry, ly*rz, ly*rw, -ly*rx

		// Compute partial result fourth column
		__m128 l_z = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 3, 3, 3)); // lz, lz, lz, lz
		__m128 r_d = _mm_shuffle_ps(r_c, r_c, _MM_SHUFFLE(2, 3, 0, 1)); // rz, ry, rx, rw
		__m128 r_l = _mm_shuffle_ps(r_k, r_k, _MM_SHUFFLE(1, 1, 0, 0)); // -, -, +, +
		__m128 r_v = _mm_xor_ps(r_d, r_l); // -rz, -ry, rx, rw
		__m128 a_z = _mm_mul_ps(l_z, r_v); // -lz*rz, -lz*ry, lz*rx, lz*rw

		// Add together partial results
		__m128 a_1 = _mm_add_ps(a_w, a_x);
		__m128 a_2 = _mm_add_ps(a_y, a_z);
		__m128 a = _mm_add_ps(a_1, a_2);

		return a;
	}


	/*--------------------------------------------------------------------
	  Quaternion Division
	--------------------------------------------------------------------*/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator/(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return lhs * Inverse(rhs);
	}
}


//========================================================================
#endif
