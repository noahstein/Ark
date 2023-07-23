/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the SSE ISA.
 * 
 * @details This file defines optimizations to the basic Quat class for 
 * use on platforms with CPUs possessing registers and instructions 
 * defined in the SSE ISA. This file contains a a class and supporting 
 * non-member functions implementing Quaternion math routines optimized 
 * for SSE. The class is then specified as a specialization of the 
 * Quat<S, I> class to be used in client code.
 * 
 * @sa ark::math::Quaternion
 * @sa Quat.h
 *
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
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
	 * @brief Quaternion with SSE Optimization
	 *
	 * @details The concept of quaternion class of a specific scalar type 
	 * optimized for the SSE ISA. Functions that implement algorithms 
	 * with SSE instructions use this concept to define their parameters.
	 *
	 * @sa QuaternionNone
	 * @sa ark::hal::simd::Sse
	 */
	template<typename Q, typename S>
	concept QuatSse = QuaternionNone<Q, S> &&
		std::derived_from<typename Q::Revision, ark::hal::simd::Sse>;


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Floating-point Quaternion
	 * 
	 * @details This class defines the data type for an SSE-optimized 
	 * single-precision floating-point quaternion. Data access member 
	 * functions model the Quaternion concept. All mathematical operations 
	 * are defined in non-member functions.
	 * 
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 *
	 * @subsubsection Operations
	 * @par
	 * <table>
	 *   <tr><th>Operation<th>Function
	 *   <tr><td>@include{doc} Math/Quaternion/NegationOperation.f
	 *     <td>@ref "operator-(QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/ConjugationOperation.f
	 *     <td>@ref "operator*(QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	 *     <td>@ref "Dot(QuatFloatSse, QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/InversionOperation.f
	 *     <td>@ref "Inverse(QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/AdditionOperation.f
	 *     <td>@ref "operator+(QuatFloatSse, QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/SubtractionOperation.f
	 *     <td>@ref "operator-(QuatFloatSse, QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatSse, QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarMultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatSse, float)"
	 *   <tr><td>@include{doc} Math/Quaternion/ScalarQuaternionMultiplicationOperation.f
	 *     <td>@ref "operator*(float, QuatFloatSse)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarDivisionOperation.f
	 *     <td>@ref "operator/(QuatFloatSse, float)"
	 * </table>
	 */
	class QuatFloatSse
	{
		/// SSE-optimized storage of 4 32-bit single-precision floats
		__m128 value_;

	public:
		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Sse;

		/// This specialization is specifically for floats.
		using Scalar = float;

		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details Like the primary template, the default constructor
		 *  leaves storage uninitialized, just like the behavior of
		 *  built-in types.
		 */
		QuatFloatSse() = default;


		/** @brief Component Constructor
		 *  @details Constructor taking the 4 quaternion components
		 *  explicitly as individual parameters.
		 */
		QuatFloatSse(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			value_ = _mm_setr_ps(w, x, y, z);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		QuatFloatSse(const Q& rhs)
			: QuatFloatSse(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief SSE Data Constructor
		 *  @details Constructor to be used by only by SSE-optimized
		 *  versions of algorithms as it uses an SSE-specific data
		 *  type. Unfortunately, there is no good way to hide it. Do not
		 *  use it in multi-platform code.
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		QuatFloatSse(__m128 value)
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


	/**
	 * @brief Specialize Quat<float, Sse> with QuatFloatSse
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Sse>
	{
		typedef QuatFloatSse type;
	};

	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Negation
	 *
	 * @details Compute a negation of a single-precision floating-point
	 * quaternion using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/Negation.txt
	 *
	 * @supersedes{QuatFloatSse, operator-(const Q&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator-(Q q) -> Q
	{
		__m128 result = _mm_sub_ps(_mm_setzero_ps(), q.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Conjugation
	 *
	 * @details Compute a conjugation of a single-precision floating-point
	 * quaternion using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 *
	 * @supersedes{QuatFloatSse, operator*(const Q&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator*(Q q) -> Q
	{
		__m128 result = _mm_move_ss((-q).SseVal(), q.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Dot Product
	 *
	 * @details Compute the dot product of two single-precision
	 * floating-point quaternions using an algorithm optimized using the 
	 * SSE ISA.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @supersedes{QuatFloatSse, Dot(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto Dot(Q lhs, Q rhs) -> float
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
	 * @brief SSE-optimized Single-precision Quaternion Addition
	 *
	 * @details Compute the addition of two single-precision floating-
	 * point quaternions using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @supersedes{QuatFloatSse, operator+(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator+(Q lhs, Q rhs) -> Q
	{
		return _mm_add_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Subtraction
	 *
	 * @details Compute the subtraction of one single-precision floating-
	 * point quaternion from another using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 *
	 * @supersedes{QuatFloatSse, operator-(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator-(Q lhs, Q rhs) -> Q
	{
		return _mm_sub_ps(lhs.SseVal(), rhs.SseVal());
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion-Scalar 
	 * Multiplication
=	 *
	 * @details Compute the product of a single-precision floating-point
	 * quaternion by a single-precision floating-point scalar value using
	 * an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 *
	 * @supersedes{QuatFloatSse,operator*(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator*(Q lhs, float rhs) -> Q
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_mul_ps(scalar, lhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Scalar-Quaternion 
	 * Multiplication
	 *
	 * @details Compute the product of a single-precision floating-point
	 * scalar value and a single-precision floating-point quaternion
	 * value using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 *
	 * @supersedes{QuatFloatSse,operator*(typename Q::Scalar\, const Q&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator*(float lhs, Q rhs) -> Q
	{
		__m128 scalar = _mm_set1_ps(lhs);
		__m128 result = _mm_mul_ps(scalar, rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Scalar Division
	 *
	 * @details Compute the quotient of a single-precision floating-point
	 * quaternion dividend by a single-precision floating-point scalar
	 * divisor using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 *
	 * @supersedes{QuatFloatSse,operator/(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	inline auto operator/(Q lhs, float rhs) -> Q
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_div_ps(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Single-precision Quaternion Multiplication
	 * 
	 * @details Compute the product of two single-precision floating-point
	 * quaternions using an SSE-optimized algorithm.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 *
	 * @supersedes{QuatFloatSse,operator*(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse<float> Q>
	auto operator*(Q lhs, Q rhs) -> Q
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
}


//************************************************************************
#endif
