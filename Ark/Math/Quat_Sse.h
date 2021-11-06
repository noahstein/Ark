/*************************************************************************
 * @file
 * @brief Quat<S, SIMD> optimizations for the SSE ISA.
 * 
 * @details This file defines opitimizations to the basic Quat class for 
 * use on platforms with CPUs possessing SSE registers and instructions.
 * Given the definition of SSE, this file more-specifically contains a 
 * specialization for Quats with float components. The original SSE 
 * definition does not include support for doubles. Optimizations include 
 * a definition of the Quat class specialized for floats that uses the 
 * SSE data type and a collection of free functions implemented against 
 * the data type using Intel itnrinsics. 
 * 
 * @sa Quat.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_SSE_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <immintrin.h>


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief SSE-optimized Quat Float Specialization
	 * 
	 * @details This class specialization defines the Quat class for SSE 
	 * and single-precision float scalars. The specialization utilites 
	 * the type __m128 to store the single-precision float information. 
	 * It is the type used extensively throughout the original SSE 
	 * intrinsics API. It represents the format of the CPU's YMM 
	 * registers. The type alos contains alignment restrictions to ensure 
	 * being able to load and store the data efficiently.
	 * 
	 * @sa Quat
	 ********************************************************************/
	template<>
	class Quat<float, ark::hal::simd::Sse>
	{
		/// SSE-optimized storage of 4 32-bit single-precision floats
		__m128 value_;

	public:
		/// This specialization is specifically for floats.
		using Scalar = float;

		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details Like the primary template, the default constructor 
		 *  leaves storage uninitialized, just like the behavior of 
		 *  built-in types.
		 */					
		Quat() = default;


		/** @brief Component Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as individaul parameters.
		 */
		Quat(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			value_ = _mm_setr_ps(w, x, y, z);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief SSE Data Constructor
		 *  @details Constructor to be used by only by SSE-optimized 
		 *  versions of algorithms as it uses an SSE-specific data 
		 *  type. Unfortunately, there is no good way to hide it. Do not 
		 *  use it in multi-platform code.
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		Quat(__m128 value)
			: value_(value)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const { return _mm_cvtss_f32(SseVal()); }
		Scalar x() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(1, 1, 1, 1))); }
		Scalar y() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(2, 2, 2, 2))); }
		Scalar z() const { return _mm_cvtss_f32(_mm_shuffle_ps(SseVal(), SseVal(), _MM_SHUFFLE(3, 3, 3, 3))); }

		/** @brief Accessor to SSE-specific data
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		__m128 SseVal() const { return value_; }	
		/// @}
	};


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Negation
	 * 
	 * @details Compute a negation of a single-precision floating-point 
	 * quaternion using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<float, ark::hal::simd::Sse> specialization. This 
	 * will supersede using the baseline QuaternionNegation expression 
	 * node when performing a negation on a Quat<float>.
	 * 
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @sa operator-(const Q& q)
	 * @sa QuaternionNegation
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		__m128 result = _mm_sub_ps(_mm_setzero_ps(), q.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Conjugation
	 * 
	 * @details Compute a conjugation of a single-precision floating-point 
	 * quaternion using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<float, ark::hal::simd::Sse> specialization. This 
	 * will supersede using the baseline QuaternionConjugate expression 
	 * node when performing a conjugation on a Quat<float>.
	 * 
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @sa operator*(const Q& q)
	 * @sa QuaternionConjugation
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		__m128 result = _mm_move_ss((-q).SseVal(), q.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Dot Product
	 * 
	 * @details Compute the dot product of two single-precision 
	 * floating-point quaternions using an SSE-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<float, ark::hal::simd::Sse> 
	 * specialization. This  will supersede using the baseline function 
	 * implementation when performing a dot product on two Quat<float> 
	 * arguments.
	 * 
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @note This implementation is superseded in SSE3.
	 * 
	 * @sa Dot(const QL& lhs, const QR& rhs)
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Inversion
	 * 
	 * @details Compute the multiplicative inverse of a single-precision 
	 * floating-point quaternion using an SSE-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<float, ark::hal::simd::Sse> 
	 * specialization. This  will supersede using the baseline function 
	 * implementation when computing an inverse of a Quat<float>.
	 * 
	 * @include{doc} Math/Quaternion/Inversion.txt
	 * 
	 * @sa Inverse(const Q& q)
	 * @sa QuaternionInversion
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto Inverse(Quat<float, SIMD> q) -> Quat<float, SIMD>
	{
		return *q / Dot(q, q);
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Addition
	 * 
	 * @details Compute the addition of two single-precision floating-
	 * point quaternions using an SSE-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<float, ark::hal::simd::Sse> 
	 * specialization. This  will supersede using the baseline function 
	 * implementation when computing the addtion of two Quat<float> 
	 * arguments.
	 * 
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @sa operator+(const QL& lhs, const QR& rhs)
	 * @sa QuaternionAddition
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator+(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return _mm_add_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Subtraction
	 * 
	 * @details Compute the subtraction of one single-precision floating-
	 * point quaternion from another using an SSE-optimized algorithm. 
	 * This implementation is selected when the HAL_SIMD parameter is set 
	 * to any SSE generation that uses the 
	 * Quat<float, ark::hal::simd::Sse> specialization. This will 
	 * supersede using the baseline function implementation when 
	 * computing the addtion of two Quat<double>arguments.
	 * 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @sa operator-(const QL& lhs, const QR& rhs)
	 * @sa QuaternionSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return _mm_sub_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Quaternion-Scalar Multiplication
	 * 
	 * @details Compute the product of a single-precision floating-point 
	 * quaternion by a single-precision floating-point scalar value using 
	 * an SSE-optimized algorithm. This implementation is selected when 
	 * the HAL_SIMD parameter is set to any SSE generation that uses the  
	 * Quat<float, ark::hal::simd::Sse> specialization. This will  
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @sa operator*(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(Quat<float, SIMD> lhs, float rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_mul_ps(scalar, lhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Scalar-Quaternion Multiplication
	 * 
	 * @details Compute the product of a single-precision floating-point 
	 * scalar value and a single-precision floating-point quaternion 
	 * value using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<float, ark::hal::simd::Sse> specialization. 
	 * This will supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @sa operator*(typename Q::Scalar s, const Q& q)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator*(float lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(lhs);
		__m128 result = _mm_mul_ps(scalar, rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Scalar Division
	 * 
	 * @details Compute the quotient of a single-precision floating-point 
	 * quaternion dividend by a single-precision floating-point scalar 
	 * divisor using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<float, ark::hal::simd::Sse> specialization. 
	 * This will supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @sa operator/(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator/(Quat<float, SIMD> lhs, float rhs) -> Quat<float, SIMD>
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_div_ps(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Multiplication
	 * 
	 * @details Compute the product of two single-precision floating-point 
	 * quaternions using an SSE-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any SSE 
	 * generation that uses the 
	 * Quat<float, ark::hal::simd::Sse> specialization. This will 
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @note This implementation superseded in SSE3.
	 * 
	 * @sa operator*(const QL& lhs, const QR& rhs)
	 * @sa QuaternionMultiplication
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief SSE-optimized Quat<float> Division
	 * 
	 * @details Compute the quotient of single-precision floating-point 
	 * quaternion dividend and divisor using an SSE-optimized algorithm. 
	 * This implementation is selected when the HAL_SIMD parameter is set 
	 * to any SSE generation that uses the 
	 * Quat<float, ark::hal::simd::Sse> specialization. This will 
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/Division.txt
	 * 
	 * @sa operator/(const QL& lhs, const QR& rhs)
	 * @sa QuaternionDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator/(Quat<float, SIMD> lhs, Quat<float, SIMD> rhs) -> Quat<float, SIMD>
	{
		return lhs * Inverse(rhs);
	}
}


//************************************************************************
#endif
