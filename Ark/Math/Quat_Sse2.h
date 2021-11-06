/*************************************************************************
 * @file
 * @brief Quat<S, SIMD> optimizations for the SSE2 ISA.
 * 
 * @details This file defines opitimizations to the basic Quat class for 
 * use on platforms with CPUs possessing SSE2 registers and instructions.
 * Given the definition of SSE2, this file more-specifically contains a 
 * specialization for Quats with double-precision floating point 
 * components. The original SSE only included single-precision floating-
 * point numerics. Consequently, the SSE optimizations do not include 
 * doubles. As no new single-precision instructions were added useful to 
 * the quaternion algorithms, the SSE2 optimizations in this file are all 
 * double-precision implementations. There is a specialization of the 
 * Quat class for double-precision components and algorithms implemented 
 * against the class specialization.
 * 
 * @sa Quat.h
 * @sa Quat_Sse.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
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
	 * @brief SSE2-optimized Quat Float Specialization
	 * 
	 * @details As the SSE2 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for single-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the original SSE 
	 * class is used as-is for SSE2.
	 * 
	 * @sa Quat<float, ark::hal::simd::Sse>
	 ********************************************************************/
	template<>
	class Quat<float, ark::hal::simd::Sse2> : public Quat<float, ark::hal::simd::Sse>
	{
		using Quat<float, ark::hal::simd::Sse>::Quat;
	};


	/*********************************************************************
	 * @brief SSE2-optimized Quat double Specialization
	 * 
	 * @details This class specialization defines the Quat class for SSE2 
	 * and double-precision float scalars. The specialization utilizes 
	 * the type __m128d to store data. As registers are still 128-bits 
	 * wide, it takes two members to store all 4 64-bit doubles 
	 * representing a quaternion. Internally, these are the same 
	 * registers used for single-precision floats in SSE; however, they 
	 * contain 2 doubles instead. 
	 * 
	 * @note This specialization superseded in AVX.
	 * 
	 * @sa Quat
	 ********************************************************************/
	template<>
	class Quat<double, ark::hal::simd::Sse2>
	{
		/// The double-precision W and X components of the quaternion
		__m128d wx_;
		/// The double-precision Y and Z components of the quaternion
		__m128d yz_;

	public:
			/// This specialization is specifically for doubles.
		using Scalar = double;

		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details Like the primary template, the default constructor 
		 *  leaves storage uninitialized, just like the behavior of 
		 *  built-in types.
		 */					
		Quat() = default;


		/** @brief Compopnent Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individaul parameters.
		 */
		Quat(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			wx_ = _mm_set_pd(x, w);
			yz_ = _mm_set_pd(z, y);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief SSE2 Data Constructor
		 *  @details Constructor to be used by only by SSE2-optimized 
		 *  versions of algorithms as it uses the SSE2-specific data 
		 *  as it is stored in an instance. Unfortunately, there is no 
		 *  good way to hide it. Do not use it in multi-platform code.
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		Quat(__m128d wx, __m128d yz)
			: wx_(wx), yz_(yz)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const { return _mm_cvtsd_f64(SseWx()); }
		Scalar x() const { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseWx(), SseWx())); }
		Scalar y() const { return _mm_cvtsd_f64 (SseYz()); }
		Scalar z() const { return _mm_cvtsd_f64(_mm_unpackhi_pd(SseYz(), SseYz())); }


		/** @brief Accessor to SSE2-specific W and X components
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		__m128d SseWx() const { return wx_; }


		/** @brief Accessor to SSE2-specific Y and Z components
		 *  @warning Only use in SSE2-specific algorithm implementations.
		 */
		__m128d SseYz() const { return yz_; }

		/// @}
	};


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Negation
	 * 
	 * @details Compute a negation of a double-precision floating-point 
	 * quaternion using an SSE2-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any SSE 
	 * generation that uses the Quat<double, ark::hal::simd::Sse2> 
	 * specialization. This will supersede using the baseline 
	 * QuaternionNegation expression node when performing a negation on a 
	 * Quat<double>.
	 * 
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator-(const Q& q)
	 * @sa QuaternionNegation
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m128d z = _mm_setzero_pd();
		__m128d wx = _mm_sub_pd(z, q.SseWx());
		__m128d yz = _mm_sub_pd(z, q.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Conjugation
	 * 
	 * @details Compute a conjugation of a double-precision floating-point 
	 * quaternion using an SSE2-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any SSE 
	 * generation that uses the Quat<double, ark::hal::simd::Sse2> 
	 * specialization. This will supersede using the baseline 
	 * QuaternionConjugate expression node when performing a conjugation 
	 * on a Quat<double>.
	 * 
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator*(const Q& q)
	 * @sa QuaternionConjugation
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Dot Product
	 * 
	 * @details Compute the dot product of two double-precision 
	 * floating-point quaternions using an SSE2-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<double, ark::hal::simd::Sse2> 
	 * specialization. This will supersede using the baseline function 
	 * implementation when performing a dot product on two Quat<double> 
	 * arguments.
	 * 
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @note This implmentation is superseded in SSE3.
	 * 
	 * @sa Dot(const QL& lhs, const QR& rhs)
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Inversion
	 * 
	 * @details Compute the multiplicative inverse of a double-precision 
	 * floating-point quaternion using an SSE2-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<double, ark::hal::simd::Sse2> 
	 * specialization. This  will supersede using the baseline function 
	 * implementation when computing an inverse of a Quat<double>.
	 * 
	 * @include{doc} Math/Quaternion/Inversion.txt
	 * 
	 * @sa Inverse(const Q& q)
	 * @sa QuaternionInversion
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto Inverse(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		return *q / Dot(q, q);
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Addition
	 * 
	 * @details Compute the addition of two double-precision floating-
	 * point quaternions using an SSE2-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any SSE generation that uses the Quat<double, ark::hal::simd::Sse2> 
	 * specialization. This  will supersede using the baseline function 
	 * implementation when computing the addtion of two Quat<double> 
	 * arguments.
	 * 
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator+(const QL& lhs, const QR& rhs)
	 * @sa QuaternionAddition
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator+(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d wx = _mm_add_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_add_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Subtraction
	 * 
	 * @details Compute the subtraction of one double-precision floating-
	 * point quaternion from another using an SSE2-optimized algorithm. 
	 * This implementation is selected when the HAL_SIMD parameter is set 
	 * to any SSE generation that uses the 
	 * Quat<double, ark::hal::simd::Sse2> specialization. This will  
	 * supersede using the baseline function implementation when computing 
	 * the addtion of two Quat<double> arguments.
	 * 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator-(const QL& lhs, const QR& rhs)
	 * @sa QuaternionSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d wx = _mm_sub_pd(lhs.SseWx(), rhs.SseWx());
		__m128d yz = _mm_sub_pd(lhs.SseYz(), rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Quaternion-Scalar Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * quaternion by a double-precision floating-point scalar value using 
	 * an SSE2-optimized algorithm. This implementation is selected when 
	 * the HAL_SIMD parameter is set to any SSE generation that uses the  
	 * Quat<double, ark::hal::simd::Sse2> specialization. This will  
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator*(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator*(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_mul_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_mul_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Scalar-Quaternion Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * scalar value and a double-precision floating-point quaternion 
	 * value using an SSE2-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<double, ark::hal::simd::Sse2> specialization. 
	 * This will supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator*(typename Q::Scalar s, const Q& q)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator*(double lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d wx = _mm_mul_pd(scalar, rhs.SseWx());
		__m128d yz = _mm_mul_pd(scalar, rhs.SseYz());
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Scalar Division
	 * 
	 * @details Compute the quotient of a double-precision floating-point 
	 * quaternion dividend by a double-precision floating-point scalar 
	 * divisor using an SSE2-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Quat<double, ark::hal::simd::Sse2> specialization. 
	 * This will supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @note This implementation is superseded in AVX.
	 * 
	 * @sa operator/(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d wx = _mm_div_pd(lhs.SseWx(), scalar);
		__m128d yz = _mm_div_pd(lhs.SseYz(), scalar);
		return {wx, yz};
	}


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Multiplication
	 * 
	 * @details Compute the product of two double-precision floating-point 
	 * quaternions using an SSE2-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any SSE 
	 * generation that uses the 
	 * Quat<double, ark::hal::simd::Sse2> specialization. This will 
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @note This implementation is superseded in SSE3.
	 * 
	 * @sa operator*(const QL& lhs, const QR& rhs)
	 * @sa QuaternionMultiplication
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief SSE2-optimized Quat<double> Division
	 * 
	 * @details Compute the quotient of double-precision floating-point 
	 * quaternion dividend and divisor using an SSE2-optimized algorithm. 
	 * This implementation is selected when the HAL_SIMD parameter is set 
	 * to any SSE generation that uses the 
	 * Quat<double, ark::hal::simd::Sse2> specialization. This will 
	 * supersede using the baseline implementation.
	 * 
	 * @include{doc} Math/Quaternion/Division.txt
	 * 
	 * @sa operator/(const QL& lhs, const QR& rhs)
	 * @sa QuaternionDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return lhs * Inverse(rhs);
	}
}


//************************************************************************
#endif
