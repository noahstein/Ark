/*************************************************************************
 * @file
 * @brief Quat<S, SIMD> optimizations for the AVX ISA.
 * 
 * @details This file contains specializations of quaternion algorithms of
 * for the AVX ISA. The major change for code from SSE4 is the expansion 
 * registers from 128- to 256-bits. AVX introduced 3-operand instructions 
 * (instead of the 2-operand instructions in earlier SSE generations); 
 * however, this isn't noticeable directly in C++ code relying upon Intel  
 * intrinsics.
 * 
 * Extending registers to 256 bits is of no benefit to single-precision 
 * floating-point quaternions; however, it's a boon to double-precision 
 * ones because now the entire quaternion fits into a single register 
 * and intrinsic data type. Consequently, this file contains a 
 * completely new implementation of quaternion algorithms for doubles.
 * 
 * @sa ark::math::Quaternion
 * @sa Quat.h
 * @sa Quat_Sse.h
 * @sa Quat_Sse2.h
 * @sa Quat_Sse3.h
 * @sa Quat_Sse4.h
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
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
	 * @brief AVX-optimized Single-precision Floating-point Quaternion
	 * 
	 * @details The AVX single-precision quaternin data type is 
	 * structurally identical to the earlier SSE generations as there is 
	 * nothing to be gained from using the full 256-bit register size. 
	 * Four floats fit in 128 bits.
	 * 
	 * @sa QuatFloatSse4
	 ********************************************************************/
	class QuatFloatAvx : public QuatFloatSse4
	{
		using QuatFloatSse4::QuatFloatSse4;
	};


	/**
	 * @brief Specialize Quat<float, Avx> with QuatFloatAvx
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Avx>
	{
		typedef QuatFloatAvx type;
	};


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Floating-point Quaternion
	 * 
	 * @details The AVX spec introduces a new 256-bit register size, so 
	 * the old SSE two-register layout is replaced with a single register 
	 * like the single-precision classes have had since the original SSE.
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
	 *     <td>@ref "operator-(QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/ConjugationOperation.f
	 *     <td>@ref "operator*(QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	 *     <td>@ref "Dot(QuatDoubleAvx, QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/InversionOperation.f
	 *     <td>@ref "Inverse(QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/AdditionOperation.f
	 *     <td>@ref "operator+(QuatDoubleAvx, QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/SubtractionOperation.f
	 *     <td>@ref "operator-(QuatDoubleAvx, QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	 *     <td>@ref "operator*(QuatDoubleAvx, QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarMultiplicationOperation.f
	 *     <td>@ref "operator*(QuatDoubleAvx, double)"
	 *   <tr><td>@include{doc} Math/Quaternion/ScalarQuaternionMultiplicationOperation.f
	 *     <td>@ref "operator*(double, QuatDoubleAvx)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarDivisionOperation.f
	 *     <td>@ref "operator/(QuatDoubleAvx, double)"
	 * </table>
	 ********************************************************************/
	class QuatDoubleAvx
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
		QuatDoubleAvx() = default;


		/** @brief Compopnent Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individaul parameters.
		 */
		QuatDoubleAvx(Scalar w, Scalar x, Scalar y, Scalar z)
		{
			value_ = _mm256_set_pd(z, y, x, w);
		}


		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		QuatDoubleAvx(const Q& rhs)
			: QuatDoubleAvx(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief AVX Data Constructor
		 *  @details Constructor to be used by only by AVX-optimized 
		 *  versions of algorithms as it uses the AVX-specific data 
		 *  as it is stored in an instance. Unfortunately, there is no 
		 *  good way to hide it. Do not use it in multi-platform code.
		 */
		QuatDoubleAvx(__m256d value)
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


	/**
	 * @brief Specialize Quat<double, Avx> with QuatDoubleAvx
	 */
	template<>
	struct QuaternionSelector<double, ark::hal::simd::Avx>
	{
		typedef QuatDoubleAvx type;
	};


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Floating-point Negation
	 * 
	 * @details Compute a negation of a double-precision floating-point 
	 * quaternion using an AVX-optimized algorithm. The new 
	 * implementation utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator-(QuatDoubleSse2)}
	 ********************************************************************/
	inline auto operator-(QuatDoubleAvx q) -> QuatDoubleAvx
	{
		__m256d zero = _mm256_setzero_pd();
		__m256d result = _mm256_sub_pd(zero, q.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion Conjugation
	 * 
	 * @details Compute a conjugation of a double-precision floating-point 
	 * quaternion using an AVX-optimized algorithm.  The new 
	 * implementation utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator*(QuatDoubleSse2)}
	 ********************************************************************/
	inline auto operator*(QuatDoubleAvx q) -> QuatDoubleAvx
	{
		__m256d val = q.AvxVal();
		__m256d neg = (-q).AvxVal();
		__m256d result = _mm256_blend_pd(val, neg, 0b1110);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion Dot Product
	 * 
	 * @details Compute the dot product of two double-precision 
	 * floating-point quaternions using an AVX-optimized algorithm. The  
	 * new implementation utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @supersedes{QuatDoubleAvx, Dot(QuatDoubleSse4\, QuatDoubleSse4)}
	 ********************************************************************/
	inline auto Dot(QuatDoubleAvx lhs, QuatDoubleAvx rhs) -> double
	{
		__m256d w_x_y_z = _mm256_mul_pd(lhs.AvxVal(), rhs.AvxVal());
		__m256d wx_yz = _mm256_hadd_pd(w_x_y_z, w_x_y_z);
		__m256d yz_wx = _mm256_permute2f128_pd(wx_yz, wx_yz, 5);
		__m256d wxyz = _mm256_add_pd(wx_yz, yz_wx);
		return _mm256_cvtsd_f64(wxyz);
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-Precision Quaternion Inversion
	 *
	 * @details Compute the multiplicative inverse of a double-precision
	 * floating-point quaternion using an AVX-optimized algorithm.
	 * @include{doc} Math/Quaternion/Inversion.txt
	 *
	 * @supersedes{QuatDoubleAvx, Inverse(QuatDoubleSse2)}
	 ********************************************************************/
	inline auto Inverse(QuatDoubleAvx q) -> QuatDoubleAvx
	{
		return *q / Dot(q, q);
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion Addition
	 * 
	 * @details Compute the addition of two double-precision floating-
	 * point quaternions using an AVX-optimized algorithm. The new 
	 * implementation utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator+(QuatDoubleSse2\, QuatDoubleSse2)}
	 ********************************************************************/
	inline auto operator+(QuatDoubleAvx lhs, QuatDoubleAvx rhs) -> QuatDoubleAvx
	{
		return _mm256_add_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion Subtraction
	 * 
	 * @details Compute the subtraction of one double-precision floating-
	 * point quaternion from another using an AVX-optimized algorithm. The 
	 * new implementation utilizes the new AVX 256-bit registers
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator-(QuatDoubleSse2\, QuatDoubleSse2)}
	 ********************************************************************/
	inline auto operator-(QuatDoubleAvx lhs, QuatDoubleAvx rhs) -> QuatDoubleAvx
	{
		return _mm256_sub_pd(lhs.AvxVal(), rhs.AvxVal());
	}


	/*********************************************************************
	 * @brief SBX-optimized Double-precision Quaternion-Scalar 
	 * Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * quaternion by a double-precision floating-point scalar value using 
	 * an AVX-optimized algorithm. The new implementation utilizes the 
	 * new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator*(QuatDoubleSse2\, double)}
	 ********************************************************************/
	inline auto operator*(QuatDoubleAvx lhs, double rhs) -> QuatDoubleAvx
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_mul_pd(scalar, lhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Scalar-Quaternion 
	 * Multiplication
	 * 
	 * @details Compute the product of a double-precision floating-point 
	 * scalar value and a double-precision floating-point quaternion 
	 * value using an AVX-optimized algorithm. The new implementation 
	 * utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator*(double\, QuatDoubleSse2)}
	 ********************************************************************/
	inline auto operator*(double lhs, QuatDoubleAvx rhs) -> QuatDoubleAvx
	{
		__m256d scalar = _mm256_set1_pd(lhs);
		__m256d result = _mm256_mul_pd(scalar, rhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion-Scalar Division
	 * 
	 * @details Compute the quotient of a double-precision floating-point 
	 * quaternion dividend by a double-precision floating-point scalar 
	 * divisor using an AVX-optimized algorithm. The new implementation 
	 * utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator/(QuatDoubleSse\, double)}
	 ********************************************************************/
	inline auto operator/(QuatDoubleAvx lhs, double rhs) -> QuatDoubleAvx
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_div_pd(lhs.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Double-precision Quaternion Multiplication
	 * 
	 * @details Compute the product of two double-precision floating-point 
	 * quaternions using an AVX-optimized algorithm.The new implementation 
	 * utilizes the new AVX 256-bit registers.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @supersedes{QuatDoubleAvx, operator*(QuatDoubleSse3\, QuatDoubleSse3)}
	 ********************************************************************/
	auto operator*(QuatDoubleAvx lhs, QuatDoubleAvx rhs) -> QuatDoubleAvx
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
