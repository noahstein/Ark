/*************************************************************************
 * @file
 * @brief Vec<S, N, SIMD> optimizations for the AVX ISA.
 * 
 * @details This file defines opitimizations to the basic Vec class for 
 * use on platforms with CPUs possessing AVX registers and instructions.
 * The AVX architecture is an externsion of SSE to extend the register 
 * size to 256 bits and includes additional instructions to support it.
 * 
 * @sa Vec.h
 * @sa Vec_Sse.h
 * @sa Vec_Sse2.h
 * @sa Vec_Sse3.h
 * @sa Vec_Sse4.h
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_AVX_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_AVX_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <concepts>

#include "Vec_Sse4.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief Avx-optimized 2-D Vec Double Specialization
	 * 
	 * @details Although the AVX specification changes the hardware 
	 * register set, the extension of registers from a width of 128 to 
	 * 256 bits has no effect on the implementation of Vec<double, 2> as 
	 * the values are only 128 bits. Thus, the for the SIMD versioning 
	 * system to work, the SSE4 class is pulled forward as-is.
	 * 
	 * @sa Vec<double, 2, ark::hal::simd::Sse4>
	 ********************************************************************/
	template<>
	class Vec<double, 2, ark::hal::simd::Avx>
		:	public Vec<double, 2, ark::hal::simd::Sse4>
	{
		using Vec<double, 2, ark::hal::simd::Sse4>::Vec;
	};


	/*********************************************************************
	 * @brief AVX-optimized 4-D Vec Float Specialization
	 * 
	 * @details Although the AVX specification changes the hardware 
	 * register set, the extension of registers from a width of 128 to 
	 * 256 bits has no effect on the implementation of Vec<float, 4> as 
	 * the values are only 128 bits. Thus, the for the SIMD versioning 
	 * system to work, the SSE4 class is pulled forward as-is.
	 * 
	 * @sa Vec<float, 4, ark::hal::simd::Sse3>
	 ********************************************************************/
	template<>
	class Vec<float, 4, ark::hal::simd::Avx>
		:	public Vec<float, 4, ark::hal::simd::Sse4>
	{
		using Vec<float, 4, ark::hal::simd::Sse4>::Vec;
	};


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief AVX-optimized 4-D Vec Double Specialization
	 * 
	 * @details This class specialization defines a 4-dimensional Vec 
	 * class the for double-precision floating-point numbers under the 
	 * AVX architecture. Thespecialization utilites the type __m1256d to 
	 * store the double-precision data. Due to SSE's 128-bit register 
	 * width, two values were needed to store 4-D double-precisions 
	 * vectors. With AVX's new 256-bit registers, 4-D double-precision 
	 * vectors may now be stored in a single register.
	 * 
	 * @sa Vec
	 ********************************************************************/
	template<>
	class Vec<double, 4, ark::hal::simd::Avx>
	{
		/// AVX-optimized storage of 4 64-bit single-precision floats
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
		Vec() = default;


		/** @brief Component Constructor
		 *  @details Constructor taking the 4 components explicitly as 
		 *  individaul parameters.
		 */
		Vec(Scalar x, Scalar y, Scalar z, Scalar w)
		{
			value_ = _mm256_setr_pd(x, y, z, w);
		}

		/** @brief Vector Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Vector concept. The Vector must also be 4-dimensional.
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar>
			&& SameDimension<Vec, V>
		Vec(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
			: Vec(static_cast<Scalar>(rhs(0)), static_cast<Scalar>(rhs(1)), static_cast<Scalar>(rhs(2)), static_cast<Scalar>(rhs(3)))
		{}

		/** @brief SSE Data Constructor
		 *  @details Constructor to be used by only by SSE-optimized 
		 *  versions of algorithms as it uses an SSE-specific data 
		 *  type. Unfortunately, there is no good way to hide it. Do not 
		 *  use it in multi-platform code.
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		Vec(__m256d value)
			: value_(value)
		{}
		/// @}

		/// This specialization is only for 4-D vectors
		static constexpr size_t Size() noexcept { return 4; }

		/// @name Accessors
		/// @{
		/** @brief Component Accessor
		 *  @param index Which component to access, beginning at 0
		 */
		Scalar operator()(size_t index) const noexcept
		{
			switch(index)
			{
				case 0:
					return _mm256_cvtsd_f64(AvxVal());

				case 1:
				{
					__m256d x = _mm256_permute_pd(AvxVal(), 1);
					return _mm256_cvtsd_f64(x);
				}

				case 2:
				{
					__m256d val = AvxVal();
					__m256d y = _mm256_permute2f128_pd(val, val, 1);
					return  _mm256_cvtsd_f64(y);
				}

				case 3:
				{
					__m256d val = AvxVal();
					__m256d yz = _mm256_permute2f128_pd(val, val, 1);
					__m256d y = _mm256_permute_pd(yz, 1);
					return _mm256_cvtsd_f64(y);
				}

				default:
					// Error condition
					return Scalar(0);
			}
		}

		/** @brief Accessor to SSE-specific data
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		__m256d AvxVal() const { return value_; }	
		/// @}
	};


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4> Negation
	 * 
	 * @details Compute a negation of a single-precision floating-point 
	 * 4-D Vec using an AVX-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any AVX generation 
	 * that uses the Vec<double, 4, ark::hal::simd::Avx> specialization. 
	 * This will supersede using the baseline VectorNegation expression 
	 * node when performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Negation4D.txt
	 * 
	 * @sa operator-(const V& v)
	 * @sa VectorNegation
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Vec<double, 4, SIMD> v) -> Vec<double, 4, SIMD>
	{
		__m256d result = _mm256_sub_pd(_mm256_setzero_pd(), v.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4> Addition
	 * 
	 * @details Compute an AVX-optimized addition of two Vec<double, 4> 
	 * vectors. This implementation is  selected when the HAL_SIMD 
	 * parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Addition4D.txt
	 * 
	 * @sa operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator+(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		__m256d result = _mm256_add_pd(vl.AvxVal(), vr.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4> Subtraction
	 * 
	 * @details Compute an AVX-optimized subtraction of one Vec<double, 4> 
	 * vector from another. This implementation is  selected when the
	 * HAL_SIMD parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Subtraction4D.txt
	 * 
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator-(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		__m256d result = _mm256_sub_pd(vl.AvxVal(), vr.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4>-Scalar Multiplication
	 * 
	 * @details Compute an AVX-optimized multiplication of a Vec<double, 4> 
	 * by a scalar. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/VectorScalarMultiplication4D.txt
	 * 
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(Vec<double, 4, SIMD> v, double s) -> Vec<double, 4, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(static_cast<double>(s));
		__m256d result = _mm256_mul_pd(v.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Scalar-Vec<double, 4> Multiplication
	 * 
	 * @details Compute an AVX-optimized multiplication of a Vec<double, 4> 
	 * by a preceding scalar. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/ScalarVectorMultiplication4D.txt
	 * 
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator*(double s, Vec<double, 4, SIMD> v) -> Vec<double, 4, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(s);
		__m256d result = _mm256_mul_pd(scalar, v.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4>-Scalar Division
	 * 
	 * @details Compute an AVX-optimized division of a Vec<double, 4> by a 
	 * scalar. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/ScalarDivision4D.txt
	 * 
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator/(Vec<double, 4, SIMD> v, double s) -> Vec<double, 4, SIMD>
	{
		__m256d scalar = _mm256_set1_pd(s);
		__m256d result = _mm256_div_pd(v.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4> Equality
	 * 
	 * @details Compute an AVX-optimized comparison of two Vec<double, 4> 
	 * vectors to each other. This implementation is selected when the
	 * HAL_SIMD parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Equality4D.txt
	 * 
	 * @sa operator==(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto operator==(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> bool
	{
		__m256d c = _mm256_cmp_pd(vl.AvxVal(), vr.AvxVal(), _CMP_EQ_OQ);
		int mask = _mm256_movemask_pd(c);
		bool result = mask == 0xf;
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized Vec<double, 4> Dot Product
	 * 
	 * @details Compute an AVX-optimized dot product of two Vec<double, 4> 
	 * vectors. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any AVX generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Avx> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/DotProduct4D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto Dot(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> double
	{
		__m256d x_y_z_w = _mm256_mul_pd(vl.AvxVal(), vr.AvxVal());
		__m256d xy_zw = _mm256_hadd_pd(x_y_z_w, x_y_z_w);
		__m256d zw_xy = _mm256_permute2f128_pd(xy_zw, xy_zw, 5);
		__m256d xyzw = _mm256_add_pd(xy_zw, zw_xy);
		return _mm256_cvtsd_f64(xyzw);
	}



	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Cross Product
	 * 
	 * @details Compute an SSE-optimized cross product of two 
	 * Vec<double, 4> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/CrossProduct4D.txt
	 * 
	 * @sa Cross(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsAvx SIMD>
	inline auto Cross(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		__m256d l = vl.AvxVal();
		__m256d r = vr.AvxVal();

		__m256d rs = _mm256_permute4x64_pd(r, _MM_SHUFFLE(3, 0, 2, 1)); // ry, rz, rx, rw
		__m256d ls = _mm256_permute4x64_pd(l, _MM_SHUFFLE(3, 0, 2, 1)); // ly, lz, lx, lw
		__m256d rl = _mm256_mul_pd(r, ls); // ly*rx, lz*ry, lx*rz, lw*rw
		__m256d a = _mm256_fmsub_pd(l, rs, rl); // lx*ry-ly*rx, ly*rz-lz*ry, lz*rx-lx*rz, 0
		__m256d result = _mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 0, 2, 1)); // ly*rz-lz*ry, lz*rx-lx*rz, lx*ry-ly*rx, 0
		return result;
	}
}


//========================================================================
#endif
