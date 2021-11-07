/*************************************************************************
 * @file
 * @brief Vec<S, N, SIMD> optimizations for the SSE ISA.
 * 
 * @details This file defines opitimizations to the basic Vec class for 
 * use on platforms with CPUs possessing SSE registers and instructions.
 * Given the definition of SSE, this file more-specifically contains a 
 * specialization for Vecs with float components. The original SSE 
 * definition does not include support for doubles. Optimizations include 
 * a definition of the Vec class specialized for floats that uses the 
 * SSE data type and a collection of free functions implemented against 
 * the data type using Intel itnrinsics. 
 * 
 * @sa Vec.h
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_SSE_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_SSE_H_INCLUDE_GUARD


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
	 * @brief SSE-optimized 4-D Vec Float Specialization
	 * 
	 * @details This class specialization defines a 4-dimensional Vec 
	 * class the for SSEand single-precision float scalars. The 
	 * specialization utilites the type __m128 to store the 
	 * single-precision float information. 
	 * It is the type used extensively throughout the original SSE 
	 * intrinsics API. It represents the format of the CPU's YMM 
	 * registers. The type alos contains alignment restrictions to ensure 
	 * being able to load and store the data efficiently.
	 * 
	 * @sa Vec
	 ********************************************************************/
	template<>
	class Vec<float, 4, ark::hal::simd::Sse>
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
		Vec() = default;


		/** @brief Component Constructor
		 *  @details Constructor taking the 4 components explicitly as 
		 *  individaul parameters.
		 */
		Vec(Scalar x, Scalar y, Scalar z, Scalar w)
		{
			value_ = _mm_setr_ps(x, y, z, w);
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
		Vec(__m128 value)
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
					return GetNth<0>(SseVal());

				case 1:
					return GetNth<1>(SseVal());

				case 2:
					return GetNth<2>(SseVal());

				case 3:
					return GetNth<3>(SseVal());

				default:
					// Error condition
					return Scalar(0);
			}
		}

		/** @brief Accessor to SSE-specific data
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		__m128 SseVal() const { return value_; }	
		/// @}

	private:
		/**
		 * @brief Get the Nth component in the SSE vector
		 * @tparam I The index of the component to read
		 * @param v The value of the vector in SSE register format
		 * @return float The value of the individual component
		 */
		template<int I> inline static auto GetNth(__m128 v) -> float
		{
			return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(I, I, I, I)));
		};
	};


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Negation
	 * 
	 * @details Compute a negation of a single-precision floating-point 
	 * 4-D Vec using an SSE-optimized algorithm. This implementation is 
	 * selected when the HAL_SIMD parameter is set to any SSE generation 
	 * that uses the Vec<float, 4, ark::hal::simd::Sse> specialization. 
	 * This will supersede using the baseline VectorNegation expression 
	 * node when performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/Negation4D.txt
	 * 
	 * @sa operator-(const V& v)
	 * @sa VectorNegation
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Vec<float, 4, SIMD> v) -> Vec<float, 4, SIMD>
	{
		__m128 result = _mm_sub_ps(_mm_setzero_ps(), v.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Addition
	 * 
	 * @details Compute an SSE-optimized addition of two Vec<float, 4> 
	 * vectors. This implementation is  selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/Addition4D.txt
	 * 
	 * @sa operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator+(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> Vec<float, 4, SIMD>
	{
		__m128 result = _mm_add_ps(vl.SseVal(), vr.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Subtraction
	 * 
	 * @details Compute an SSE-optimized subtraction of one Vec<float, 4> 
	 * vector from another. This implementation is  selected when the
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/Subtraction4D.txt
	 * 
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator-(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> Vec<float, 4, SIMD>
	{
		__m128 result = _mm_sub_ps(vl.SseVal(), vr.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4>-Scalar Multiplication
	 * 
	 * @details Compute an SSE-optimized multiplication of a Vec<float, 4> 
	 * by a scalar. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/VectorScalarMultiplication4D.txt
	 * 
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD, typename S>
		requires std::is_convertible_v<S, float>
	inline auto operator*(Vec<float, 4, SIMD> v, const S& s) -> Vec<float, 4, SIMD>
	{
		__m128 scalar = _mm_set1_ps(static_cast<float>(s));
		__m128 result = _mm_mul_ps(v.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Scalar-Vec<float, 4> Multiplication
	 * 
	 * @details Compute an SSE-optimized multiplication of a Vec<float, 4> 
	 * by a preceding scalar. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/ScalarVectorMultiplication4D.txt
	 * 
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD, typename S>
		requires std::is_convertible_v<S, float>
	inline auto operator*(const S& s, Vec<float, 4, SIMD> v) -> Vec<float, 4, SIMD>
	{
		__m128 scalar = _mm_set1_ps(static_cast<float>(s));
		__m128 result = _mm_mul_ps(scalar, v.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4>-Scalar Division
	 * 
	 * @details Compute an SSE-optimized division of a Vec<float, 4> by a 
	 * scalar. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/ScalarDivision4D.txt
	 * 
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD, typename S>
		requires std::is_convertible_v<S, float>
	inline auto operator/(Vec<float, 4, SIMD> v, const S& s) -> Vec<float, 4, SIMD>
	{
		__m128 scalar = _mm_set1_ps(static_cast<float>(s));
		__m128 result = _mm_div_ps(v.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Equality
	 * 
	 * @details Compute an SSE-optimized comparison of two Vec<float, 4> 
	 * vectors to each other. This implementation is selected when the
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/Equality4D.txt
	 * 
	 * @sa operator==(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto operator==(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> bool
	{
		__m128 c = _mm_cmpeq_ps(vl.SseVal(), vr.SseVal());
		int mask = _mm_movemask_ps(c);
		bool result = mask == 0xf;
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Dot Product
	 * 
	 * @details Compute an SSE-optimized dot product of two Vec<float, 4> 
	 * vectors. This implementation is selected when the HAL_SIMD 
	 * parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/DotProduct4D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto Dot(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> float
	{
		__m128 m = _mm_mul_ps(vl.SseVal(), vr.SseVal()); // lxrx, lyry, lzrz, lwrw
		__m128 s = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 1, 2, 3)); // lwrw, lzrz, lyry, lxrx
		__m128 m2 = _mm_add_ps(m, s); // lxrx+lwrw, lyry+lzrz, lyry+lzrz, lxrx+lwrw
		__m128 s2 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1, 0, 3, 2)); // lyry+lzrz, lxrx+lwrw, lyry+lzrz, lxrx+lwrw
		__m128 m3 = _mm_add_ps(m2, s2); // lxrx+lyry+lzrz+lwrw, ...
		float result = _mm_cvtss_f32(m3);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized Vec<float, 4> Cross Product
	 * 
	 * @details Compute an SSE-optimized cross product of two 
	 * Vec<float, 4> vectors. This implementation is selected when the 
	 * HAL_SIMD parameter is set to any SSE generation that uses the 
	 * Vec<float, 4, ark::hal::simd::Sse> specialization.  This will 
	 * supersede using the baseline VectorNegation expression node when 
	 * performing a negation on a Vec<float, 4>.
	 * 
	 * @include{doc} Math/Vector/CrossProduct4D.txt
	 * 
	 * @sa Cross(const V& vl, const V& vr)
	 ********************************************************************/
	template<ark::hal::simd::IsSse SIMD>
	inline auto Cross(Vec<float, 4, SIMD> vl, Vec<float, 4, SIMD> vr) -> Vec<float, 4, SIMD>
	{
		__m128 l = vl.SseVal();
		__m128 r = vr.SseVal();

		__m128 l1 = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 0, 2, 1)); // ly, lz, lx
		__m128 r1 = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 1, 0, 2)); // rz, rx, ry
		__m128 f1 = _mm_mul_ps(l1, r1); // ly*rz, lz*rx, lx*ry

		__m128 l2 = _mm_shuffle_ps(l, l, _MM_SHUFFLE(0, 1, 0, 2)); // lz, lx, ly
		__m128 r2 = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 2, 1)); // ry, rz, rx
		__m128 f2 = _mm_mul_ps(l2, r2); // lz*ry, lx*rz, ly*rx

		__m128 result = _mm_sub_ps(f1, f2); // ly*rz-lz*ry, lz*rx-lx*rz, lx*ry-ly*rx
		return result;
	}
}


//========================================================================
#endif
