/*************************************************************************
 * @file
 * @brief Vec<S, N, SIMD> optimizations for the SSE2 ISA.
 * 
 * @details This file defines opitimizations to the basic Vec class for 
 * use on platforms with CPUs possessing SSE2 registers and instructions.
 * Given the definition of SSE2, this file more-specifically contains a 
 * specialization for Vecs with double-precision floating point 
 * components. The original SSE only included single-precision floating-
 * point numerics. Consequently, the SSE optimizations do not include 
 * doubles. As no new single-precision instructions were added useful to 
 * the vector algorithms, the SSE2 optimizations in this file are all 
 * double-precision implementations. There is a specialization of the 
 * Vec class for double-precision components and algorithms implemented 
 * against the class specialization.
 * 
 * @sa Vec.h
 * @sa Vec_Sse.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_SSE2_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_SSE2_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Vec_Sse.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief SSE2-optimized 4-D Vec Float Specialization
	 * 
	 * @details As the SSE2 specification does not change the hardware 
	 * register set, the Intel intrinsic type representing the register 
	 * format has not changed for single-precision floating-point 
	 * vectors. For the SIMD versioning system to work, the original SSE 
	 * class is used as-is for SSE2.
	 * 
	 * @sa Vec<float, ark::hal::simd::Sse>
	 ********************************************************************/
	template<>
	class Vec<float, 4, ark::hal::simd::Sse2> : public Vec<float, 4, ark::hal::simd::Sse>
	{
		using Vec<float, 4, ark::hal::simd::Sse>::Vec;
	};

	/*********************************************************************
	 * @brief SSE2-optimized 4-D Vec Double Specialization
	 * 
	 * @details This class specialization defines the Vec class for SSE2 
	 * and double-precision floating-point scalars. The specialization 
	 * utilizes the type __m128d to store data. As registers are still 
	 * 128-bits wide, it takes two members to store all 4 64-bit doubles 
	 * representing a 4-D vector. Internally, these are the same 
	 * registers used for single-precision floats in SSE; however, they 
	 * contain 2 doubles instead of 4 singles. 
	 * 
	 * @note This specialization superseded in AVX.
	 * 
	 * @sa Vec
	 ********************************************************************/
	template<>
	class Vec<double, 4, ark::hal::simd::Sse2>
	{
		/// The double-precision components at indices 0 and 1
		__m128d v01_;
		/// The double-precision components at indices 2 and 3
		__m128d v23_;

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
			v01_ = _mm_setr_pd(x, y);
			v23_ = _mm_setr_pd(z, w);
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
		Vec(__m128d v01, __m128d v23)
			: v01_(v01)
			, v23_(v23)
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
					return _mm_cvtsd_f64(Sse01());

				case 1:
					return _mm_cvtsd_f64(_mm_unpackhi_pd(Sse01(), Sse01()));

				case 2:
					return _mm_cvtsd_f64(Sse23());

				case 3:
					return _mm_cvtsd_f64(_mm_unpackhi_pd(Sse23(), Sse23()));

				default:
					// Error condition
					return Scalar(0);
			}
		}

		/** @brief Accessor to SSE-specific data of first two components
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		__m128d Sse01() const { return v01_; }	

		/** @brief Accessor to SSE-specific data of last two components
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		__m128d Sse23() const { return v23_; }	
		/// @}
	};

	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Negation
	 * 
	 * @details Compute a negation of a double-precision floating-point 
	 * 4-D Vec using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Vec<double, 4, ark::hal::simd::Sse2> specialization. 
	 * This will supersede using the baseline VectorNegation expression 
	 * node when performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Negation.txt
	 * 
	 * @sa operator-(const V& v)
	 * @sa VectorNegation
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Vec<double, 4, SIMD> v) -> Vec<double, 4, SIMD>
	{
		__m128d zero = _mm_setzero_pd();
		__m128d v01 = _mm_sub_pd(zero, v.Sse01());
		__m128d v23 = _mm_sub_pd(zero, v.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Addition
	 * 
	 * @details Compute an SSE-optimized addition of two Vec<double, 4> 
	 * vectors. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse2> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Addition.txt
	 * 
	 * @sa operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator+(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		__m128d v01 = _mm_add_pd(vl.Sse01(), vr.Sse01());
		__m128d v23 = _mm_add_pd(vl.Sse23(), vr.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Subtraction
	 * 
	 * @details Compute an SSE-optimized subtraction of one 
	 * Vec<double, 4> vector from another. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Vec<double, 4, ark::hal::simd::Sse> specialization.  
	 * This will supersede using the baseline VectorNegation expression 
	 * node when performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Subtraction.txt
	 * 
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator-(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		__m128d v01 = _mm_sub_pd(vl.Sse01(), vr.Sse01());
		__m128d v23 = _mm_sub_pd(vl.Sse23(), vr.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4>-Scalar Multiplication
	 * 
	 * @details Compute an SSE-optimized multiplication of a 
	 * Vec<double, 4> by a scalar. This implementation is selected when 
	 * the HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse> specialization. This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/VectorScalarMultiplication.txt
	 * 
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD, typename S>
		requires std::is_convertible_v<S, double>
	inline auto operator*(Vec<double, 4, SIMD> v, const S& s) -> Vec<double, 4, SIMD>
	{
		__m128d scalar = _mm_set1_pd(static_cast<double>(s));
		__m128d v01 = _mm_mul_pd(v.Sse01(), scalar);
		__m128d v23 = _mm_mul_pd(v.Sse23(), scalar);
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Scalar-Vec<double, 4> Multiplication
	 * 
	 * @details Compute an SSE-optimized multiplication of a 
	 * Vec<double, 4> by a preceding scalar. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Vec<double, 4, ark::hal::simd::Sse> specialization. 
	 * This will supersede using the baseline VectorNegation expression 
	 * node when performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/ScalarVectorMultiplication.txt
	 * 
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD, typename S>
		requires std::is_convertible_v<S, double>
	inline auto operator*(const S& s, Vec<double, 4, SIMD> v) -> Vec<double, 4, SIMD>
	{
		__m128d scalar = _mm_set1_pd(static_cast<double>(s));
		__m128d v01 = _mm_mul_pd(scalar, v.Sse01());
		__m128d v23 = _mm_mul_pd(scalar, v.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4>-Scalar Division
	 * 
	 * @details Compute an SSE-optimized division of a Vec<double, 4> by 
	 * a scalar. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/VectorScalarDivision.txt
	 * 
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD, typename S>
		requires std::is_convertible_v<S, double>
	inline auto operator/(Vec<double, 4, SIMD> v, const S& s) -> Vec<double, 4, SIMD>
	{
		__m128d scalar = _mm_set1_pd(static_cast<double>(s));
		__m128d v01 = _mm_div_pd(v.Sse01(), scalar);
		__m128d v23 = _mm_div_pd(v.Sse23(), scalar);
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Equality
	 * 
	 * @details Compute an SSE-optimized comparison of two Vec<double, 4> 
	 * vectors to each other. This implementation is selected when the
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/Comparison.txt
	 * 
	 * @sa operator==(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto operator==(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> bool
	{
		__m128d c01 = _mm_cmpeq_pd(vl.Sse01(), vr.Sse01());
		int m01 = _mm_movemask_pd(c01);
		__m128d c23 = _mm_cmpeq_pd(vl.Sse23(), vr.Sse23());
		int m23 = _mm_movemask_pd(c23);
		bool mask = (m01 & m23) == 0x3;
		return mask;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<double, 4> Dot Product
	 * 
	 * @details Compute an SSE-optimized dot product of two Vec<double, 4> 
	 * vectors. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<double, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<double, 4>.
	 * 
	 * @include{doc} Math/Vector/DotProduct.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto Dot(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> double
	{
		__m128d v01 = _mm_mul_pd(vl.Sse01(), vr.Sse01());
		__m128d v10 = _mm_shuffle_pd(v01, v01, _MM_SHUFFLE2(0, 1));
		__m128d va = _mm_add_pd(v01, v10);

		__m128d v23 = _mm_mul_pd(vl.Sse23(), vr.Sse23());
		__m128d v32 = _mm_shuffle_pd(v23, v23, _MM_SHUFFLE2(0, 1));
		__m128d vb = _mm_add_pd(v23, v32);

		__m128d dp = _mm_add_pd(va, vb);
		double result = _mm_cvtsd_f64(dp);
		return result;
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
	template<ark::hal::simd::IsSse2 SIMD>
	inline auto Cross(Vec<double, 4, SIMD> vl, Vec<double, 4, SIMD> vr) -> Vec<double, 4, SIMD>
	{
		// Gather Data
		__m128d l01 = vl.Sse01();
		__m128d l23 = vl.Sse23();
		__m128d r01 = vr.Sse01();
		__m128d r23 = vr.Sse23();
		__m128d zzz = _mm_setzero_pd();

		// Compute first two components
		__m128d l12 = _mm_shuffle_pd(l01, l23, _MM_SHUFFLE2(0, 1)); // l1, l2
		__m128d r20 = _mm_shuffle_pd(r23, r01, _MM_SHUFFLE2(0, 0)); // r2, r0
		__m128d c0a = _mm_mul_pd(l12, r20); // l1r2, l2r0

		__m128d l20 = _mm_shuffle_pd(l23, l01, _MM_SHUFFLE2(0, 0)); // l2, l0
		__m128d r12 = _mm_shuffle_pd(r01, r23, _MM_SHUFFLE2(0, 1)); // r1, r2
		__m128d c0b = _mm_mul_pd(l20, r12); // l2r1, l0r2

		__m128d v01 = _mm_sub_pd(c0a, c0b); // l1r2-l2r1, l2r0-l0r2

		// Compute last component
		__m128d l0z = _mm_shuffle_pd(l01, zzz, _MM_SHUFFLE2(0, 0)); // l0, 0
		__m128d r1z = _mm_shuffle_pd(r01, zzz, _MM_SHUFFLE2(0, 1)); // r1, 0
		__m128d c1a = _mm_mul_pd(l0z, r1z); // l0r1, 0

		__m128d l1z = _mm_shuffle_pd(l01, zzz, _MM_SHUFFLE2(0, 1)); // l1, 0
		__m128d r0z = _mm_shuffle_pd(r01, zzz, _MM_SHUFFLE2(0, 0)); // r0, 0
		__m128d c1b = _mm_mul_pd(l1z, r0z); // l1r0, 0

		__m128d v23 = _mm_sub_pd(c1a, c1b); // l0r1-l1r0, 0

		// Final result
		return {v01, v23};
	}
}


//========================================================================
#endif
