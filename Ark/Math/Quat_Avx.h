/*************************************************************************
 * @file
 * @brief Quat<S, SIMD> optimizations for the AVX ISA.
 * 
 * @details This file contains specializations of Quat algorithms for the 
 * AVX ISA. The major change for code from SSE4 is the expansion of 
 * registers from 128- to 256-bits. AVX introduced 3-operand instructions 
 * (instead of the 2-operand instructions in earlier SSE generations); 
 * however, this isn't noticeable directly in C++ code relying upon INtel  
 * intrinsics.
 * 
 * Extending registers to 256 bits is of no benefit to single-precision 
 * floating-point quaternions; however, it's a boon to double-precision 
 * ones because now the entire quaternion fits into a single register 
 * and intrinsic data type. Consequently, this file contains a 
 * completely new implementation of quaternion algorithms for doubles.
 * 
 * @sa Quat.h
 * @sa Quat_Sse.h
 * @sa Quat_Sse2.h
 * @sa Quat_Sse3.h
 * @sa Quat_Sse4.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_AVX_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_AVX_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Quat_Sse4.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief AVX-optimized Quat Float Specialization
	 * 
	 * @details The AVX Quat<float> data type is structurally identical 
	 * to the earlier SSE one as there is nothing to be gained from using 
	 * the full 256-bit register size. Four floats fit in 128 bits.
	 * 
	 * @sa Quat<float, ark::hal::simd::Sse4>
	 ********************************************************************/
	template<>
	class Quat<float, ark::hal::simd::Avx> : public Quat<float, ark::hal::simd::Sse4>
	{
		using Quat<float, ark::hal::simd::Sse4>::Quat;
	};


	/*********************************************************************
	 * @brief SSE4-optimized Quat Double Specialization
	 * 
	 * @details The AVX spec introduces a new 256-bit register size, so 
	 * the old SSE two-register layout is replaced with a single register 
	 * like the float specialization has had since the original SSE.
	 * 
	 * @sa Quat<double, ark::hal::simd::Sse4>
	 ********************************************************************/
	template<> class Quat<double, ark::hal::simd::Avx>
	{
		/// All four double-precision components: w, x, y, and z
		__m256d value_;

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
			value_ = _mm256_set_pd(z, y, x, w);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		Quat(const Q& rhs)
			: Quat(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief AVX Data Constructor
		 *  @details Constructor to be used by only by AVX-optimized 
		 *  versions of algorithms as it uses the AVX-specific data 
		 *  as it is stored in an instance. Unfortunately, there is no 
		 *  good way to hide it. Do not use it in multi-platform code.
		 */
		Quat(__m256d value)
			: value_(value)
		{}
		/// @}

		/// @name Accessors
		/// @{
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

		/** @brief Accessor to AVX-specific data
		 *  @warning Only use in AVX-specific algorithm implementations.
		 */
		__m256d AvxVal() const { return value_; }
		/// @}
	};


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Negation
	 * 
	 * @details Compute a negation of a double-precision floating-point 
	 * quaternion using an SSE2-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any AVX 
	 * generation that uses the Quat<double, ark::hal::simd::Avx> 
	 * specialization. This supersedes the earlier SSE implementation and 
	 * is mandatory due to the new data format. 
	 * 
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @sa operator-(const Q& q)
	 * @sa QuaternionNegation
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m256d zero = _mm256_setzero_pd();
		__m256d result = _mm256_sub_pd(zero, q.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Conjugation
	 * 
	 * @details Compute a conjugation of a double-precision floating-point 
	 * quaternion using an AVX-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any AVX 
	 * generation that uses the Quat<double, ark::hal::simd::Avx> 
	 * specialization. This supersedes the earlier SSE implementation and 
	 * is mandatory due to the new data format. 
	 * 
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @sa operator*(const Q& q)
	 * @sa QuaternionConjugation
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(Quat<double, SIMD> q) -> Quat<double, SIMD>
	{
		__m256d val = q.AvxVal();
		__m256d neg = (-q).AvxVal();
		__m256d result = _mm256_blend_pd(val, neg, 0b1110);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Dot Product
	 * 
	 * @details Compute the dot product of two double-precision 
	 * floating-point quaternions using an AVX-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any AVX generation that uses the Quat<double, ark::hal::simd::Avx> 
	 * specialization. This supersedes the earlier SSE implementation and 
	 * is mandatory due to the new data format. 
	 * 
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @sa Dot(const QL& lhs, const QR& rhs)
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto Dot(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> double
	{
		__m256d w_x_y_z = _mm256_mul_pd(lhs.AvxVal(), rhs.AvxVal());
		__m256d wx_yz = _mm256_hadd_pd(w_x_y_z, w_x_y_z);
		__m256d yz_wx = _mm256_permute2f128_pd(wx_yz, wx_yz, 5);
		__m256d wxyz = _mm256_add_pd(wx_yz, yz_wx);
		return _mm256_cvtsd_f64(wxyz);
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Addition
	 * 
	 * @details Compute the addition of two double-precision floating-
	 * point quaternions using an AVX-optimized algorithm. This 
	 * implementation is selected when the HAL_SIMD parameter is set to 
	 * any AVX generation that uses the Quat<double, ark::hal::simd::Avx> 
	 * specialization. This supersedes the earlier SSE implementation and 
	 * is mandatory due to the new data format. 
	 * 
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @sa operator+(const QL& lhs, const QR& rhs)
	 * @sa QuaternionAddition
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator+(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return _mm256_add_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Subtraction
	 * 
	 * @details Compute the subtraction of one double-precision floating-
	 * point quaternion from another using an AVX-optimized algorithm. 
	 * This implementation is selected when the HAL_SIMD parameter is set 
	 * to any AVX generation that uses the 
	 * Quat<double, ark::hal::simd::Avx> specialization. This supersedes 
	 * the earlier SSE implementation and is mandatory due to the new 
	 * data format. 
	 * 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @sa operator-(const QL& lhs, const QR& rhs)
	 * @sa QuaternionSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Quat<double, SIMD> lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		return _mm256_sub_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*********************************************************************
	 * @brief SBX-optimized Quat<double> Quaternion-Scalar Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * quaternion by a double-precision floating-point scalar value using 
	 * an AVX-optimized algorithm. This implementation is selected when 
	 * the HAL_SIMD parameter is set to any AVX generation that uses the  
	 * Quat<double, ark::hal::simd::Avx> specialization. This supersedes 
	 * the earlier SSE implementation and is mandatory due to the new 
	 * data format. 
	 * 
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @sa operator*(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_mul_pd(scalar, lhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Scalar-Quaternion Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * scalar value and a double-precision floating-point quaternion 
	 * value using an AVX-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any AVX generation 
	 * that uses the Quat<double, ark::hal::simd::Avx> specialization.
	 * This supersedes the earlier SSE implementation and is mandatory 
	 * due to the new data format.
	 *  
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @sa operator*(typename Q::Scalar s, const Q& q)
	 * @sa QuaternionScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(double lhs, Quat<double, SIMD> rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(lhs);
		__m256d result = _mm256_mul_pd(scalar, rhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Scalar Division
	 * 
	 * @details Compute the quotient of a double-precision floating-point 
	 * quaternion dividend by a double-precision floating-point scalar 
	 * divisor using an AVX-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any AVX generation 
	 * that uses the Quat<double, ark::hal::simd::AVx> specialization. 
	 * This supersedes the earlier SSE implementation and is mandatory 
	 * due to the new data format.
	 * 
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @sa operator/(const Q& q, typename Q::Scalar s)
	 * @sa QuaternionScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator/(Quat<double, SIMD> lhs, double rhs) -> Quat<double, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_div_pd(lhs.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Quat<double> Multiplication
	 * 
	 * @details Compute the product of two double-precision floating-point 
	 * quaternions using an AVX-optimized algorithm. This implementation 
	 * is selected when the HAL_SIMD parameter is set to any AVX 
	 * generation that uses the 
	 * Quat<double, ark::hal::simd::Avx> specialization. This supersedes 
	 * the earlier SSE implementation and is mandatory due to the new 
	 * data format.
	 * 
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @note This implementation is superseded in AVX2.
	 * 
	 * @sa operator*(const QL& lhs, const QR& rhs)
	 * @sa QuaternionMultiplication
	 ********************************************************************/
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


//************************************************************************
#endif
