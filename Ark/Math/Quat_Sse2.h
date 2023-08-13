/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the SSE2 ISA.
 * 
 * @details This file defines optimizations to the basic Quat class for
 * use on platforms with CPUs possessing registers and instructions
 * defined in the SSE2 ISA. The data representation and operations of 
 * single-precision floating-point quaternions remains unchanged from 
 * the original SSE implementations. SSE2 introduces a register format 
 * and instructions for the efficient implementation of double-precision 
 * floating-point quaternions.
 *
 * @sa ark::math::Quaternion
 * @sa Quat.h
 * @sa Quat_Sse.h
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_SSE2_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Quat_Sse.h"


//************************************************************************
// Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief Quaternion with SSE2 Optimization
	 *
	 * @details The concept of quaternion class of a specific scalar type
	 * optimized for the SSE2 ISA.
	 *
	 * @sa QuaternionSse
	 * @sa ark::hal::simd::Sse2
	 */
	template<typename Q, typename S>
	concept QuatSse2 = QuatSse<Q, S> &&
		std::derived_from<typename Q::Revision, ark::hal::simd::Sse2>;


	/*********************************************************************
	 * @brief SSE2-optimized Single-precision Floating-point Quaternion
	 * 
	 * @details The SSE2 ISA specification contains no change to the 
	 * register set affecting single-precision floating-point operations; 
	 * therefore, the SSE class is pulled forward.
	 *
	 * It is not necessary to define this class. The Ark architecture 
	 * permits directly specializing on the original SSE implementation 
	 * when there are no new superseding operations defined for a SIMD 
	 * generation. The choice to pull the original SSE specification 
	 * forward and define it as the SSE2 specification is for 
	 * consistency. This way, the SSE3 specification can easily just 
	 * derive from SSE2 instead of the programmer needing to look through 
	 * files to find the where the last register layout was specified.
	 * 
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" Quaternionn
	 * 
	 * @sa QuatFloatSse
	 ********************************************************************/
	class QuatFloatSse2 : public QuatFloatSse
	{
	public:
		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Sse2;

		using QuatFloatSse::QuatFloatSse;
	};


	/**
	 * @brief Specialize Quat<float, Sse2> with QuatFloatSse2
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Sse2>
	{
		typedef QuatFloatSse2 type;
	};


	/*********************************************************************
	* @brief SSE2-optimized Double-precision Floating-point Quaternion
	*
	* @details This class defines the data type for an SSE2-optimized
	* double-precision floating-point quaternion. Data access member
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
	*	<tr><th>Operation<th>Function
	*   <tr><td>@include{doc} Math/Quaternion/NegationOperation.f
	*     <td>@ref "operator-(QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/ConjugationOperation.f
	*     <td>@ref "operator*(QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	*     <td>@ref "Dot(QuatDoubleSse2, QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/InversionOperation.f
	*     <td>@ref "Inverse(QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/AdditionOperation.f
	*     <td>@ref "operator+(QuatDoubleSse2, QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/SubtractionOperation.f
	*     <td>@ref "operator-(QuatDoubleSse2, QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	*     <td>@ref "operator*(QuatDoubleSse2, QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarMultiplicationOperation.f
	*     <td>@ref "operator*(QuatDoubleSse2, double)"
	*   <tr><td>@include{doc} Math/Quaternion/ScalarQuaternionMultiplicationOperation.f
	*     <td>@ref "operator*(double, QuatDoubleSse2)"
	*   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarDivisionOperation.f
	*     <td>@ref "operator/(QuatDoubleSse2, double)"
	* </table>
	*/
	class QuatDoubleSse2
	{
		/// @name Data
		/// @{

		/// The double-precision W and X components of the quaternion
		__m128d wx_;
		/// The double-precision Y and Z components of the quaternion
		__m128d yz_;

		/// @}

	public:
		/// @name Configuration Types
		/// @{

		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Sse2;

		/// This specialization is specifically for doubles.
		using Scalar = double;

		/// @}

		/// @name noexcept Tests
		/// @{

		/** @brief Can a conversion from Q class's Scalar be nothrow converted to Scalar?
		 *  @tparam Q The type of the quaternion being copied from
		 *  @details If another quaternion's Scalar values may be nothrow
		 *  converted to this QuaternBasic's Scalar type then the copy
		 *  is also nothrow as it is only composed of 4 Scalar instances.
		 */
		template<typename Q>
		constexpr static bool IsNoThrowConvertible = std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>;

		/// @}

		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details Like the primary template, the default constructor 
		 *  leaves storage uninitialized, just like the behavior of 
		 *  built-in types.
		 */					
		QuatDoubleSse2() noexcept = default;


		/** @brief Component Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individual parameters.
		 */
		QuatDoubleSse2(Scalar w, Scalar x, Scalar y, Scalar z) noexcept
		{
			wx_ = _mm_set_pd(x, w);
			yz_ = _mm_set_pd(z, y);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		QuatDoubleSse2(const Q& rhs) noexcept(IsNoThrowConvertible<typename Q::Scalar>)
			: QuatDoubleSse2(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief SSE2 Data Constructor
		 *  @details Constructor to be used by only by SSE2-optimized 
		 *  versions of algorithms as it uses the SSE2-specific data 
		 *  as it is stored in an instance. Unfortunately, there is no 
		 *  good way to hide it. Do not use it in multi-platform code.
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		QuatDoubleSse2(__m128d wx, __m128d yz) noexcept
			: wx_(wx), yz_(yz)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return _mm_cvtsd_f64(SseWx()); }
		Scalar x() const noexcept { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseWx(), SseWx())); }
		Scalar y() const noexcept { return _mm_cvtsd_f64 (SseYz()); }
		Scalar z() const noexcept { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseYz(), SseYz())); }


		/** @brief Accessor to SSE2-specific W and X components
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		__m128d SseWx() const noexcept { return wx_; }


		/** @brief Accessor to SSE2-specific Y and Z components
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		__m128d SseYz() const noexcept { return yz_; }

		/// @}
	};


	/**
	 * @brief Specialize Quat<double, Sse2> with QuatDoubleSse2
	 */
	template<>
	struct QuaternionSelector<double, ark::hal::simd::Sse2>
	{
		typedef QuatDoubleSse2 type;
	};


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion Negation
	 * 
	 * @details Compute a negation of a double-precision floating-point 
	 * quaternion using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator-(const Q&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator-(Q q) noexcept -> Q
	{
		__m128d z = _mm_setzero_pd();
		__m128d wx = _mm_sub_pd(z, q.SseWx());
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion Conjugation
	 * 
	 * @details Compute a conjugation of a double-precision floating-point 
	 * quaternion using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator*(const Q&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator*(Q q) noexcept -> Q
	{
		__m128d z = _mm_setzero_pd();
		__m128d wxi = q.SseWx();
		__m128d wxn = _mm_sub_pd(z, wxi);
		__m128d wx = _mm_move_sd(wxn, wxi);
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-Precision Quaternion Dot Product
	 * 
	 * @details Compute the dot product of two double-precision 
	 * floating-point quaternions using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @supersedes{QuatDoubleSse2, Dot(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto Dot(QuatDoubleSse2 lhs, QuatDoubleSse2 rhs) noexcept -> double
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


	/*********************************************************************
	 * @brief SSE2-optimized Double-Precision Quaternion Addition
	 * 
	 * @details Compute the addition of two double-precision floating-
	 * point quaternions using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator+(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator+(Q lhs, Q rhs) noexcept -> Q
	{
		__m128d wx = _mm_add_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_add_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion Subtraction
	 * 
	 * @details Compute the subtraction of one double-precision floating-
	 * point quaternion from another using an SSE2-optimized algorithm. 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator-(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator-(Q lhs, Q rhs) noexcept -> Q
	{
		__m128d wx = _mm_sub_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_sub_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion-Scalar 
	 * Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * quaternion by a double-precision floating-point scalar value using 
	 * an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator*(const Q&\, typename Q::Scalar s)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator*(Q lhs, double rhs) noexcept -> Q
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_mul_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_mul_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Scalar-Quaternion 
	 * Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * scalar value and a double-precision floating-point quaternion 
	 * value using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator*(typename Q::Scalar\, const Q&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator*(double lhs, Q rhs) noexcept -> Q
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d wx = _mm_mul_pd(scalar, rhs.SseWx());
		__m128d yz = _mm_mul_pd(scalar, rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion-Scalar Division
	 * 
	 * @details Compute the quotient of a double-precision floating-point 
	 * quaternion dividend by a double-precision floating-point scalar 
	 * divisor using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @supersedes{QuatDoubleSse2,operator/(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	inline auto operator/(Q lhs, double rhs) noexcept -> Q
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_div_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_div_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Double-precision Quaternion Multiplication
	 * 
	 * @details Compute the product of two double-precision floating-point 
	 * quaternions using an SSE2-optimized algorithm.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @supersedes{QuatDoubleSse2, operator*(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatSse2<double> Q>
	auto operator*(Q lhs, Q rhs) noexcept -> Q
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
}


//************************************************************************
#endif
