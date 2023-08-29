 /*************************************************************************
  * @file
  * @brief SSE2-optimized Vec Implementation
  *
  * @details This file defines optimizations to the Vec type family
  * utilizing the SSE2 SIMD ISA. The original SSE spec did not include 
  * support for double-precision floating-point values. Intel added 
  * such support in SSE2. It did not, however, increase the size of the 
  * register set. Registers remain 128 bits wide. Thus, each register 
  * holds only two doubles. Consequently, there is not only an optimized 
  * 4-D double-precision vector but also a 2-D one; therefore, this 
  * file defines two optimized SSE2 classes:
  *
  * - VecDouble2Sse2: 2-D double-precision vector (Vec<double, 2, Sse2>)
  * - VecDouble4Sse2: 4-D double-precision vector (Vec<double, 4, Sse2>)
  *
  * @sa Vec.h
  * @se Vec_Sse.h
  * @sa ::ark::hal::simd::Sse2
  *
  * @author Noah Stein
  * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
  ************************************************************************/

#if !defined(ARK_MATH_VEC_SSE2_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_SSE2_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <concepts>

#include "Vec_Sse.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief SSE2-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 *
	 * @details Concept for declaring parameters of SSE2-optimized
	 * functions.It ensures the parameter is of a type optimized for the
	 * SSE2 SIMD ISA but not the original SSE.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 *
	 * @sa ark::math::VecSse
	 * @sa ark::hal::simd::Sse2Family
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecSse2 = VecSse<V, S, N> &&
		::ark::hal::simd::Sse2Family<typename V::Revision>;


	//====================================================================
	// 4-D Single-precision Vector
	//====================================================================

	/*********************************************************************
	 * @brief SSE2-optimized 4-D Single-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE to SSE2 do not result in a new 
	 * register specification applicable to further 4-D single-precision 
	 * floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse2Family
	 */
	class VecFloat4Sse2 : public VecFloat4Sse
	{
	public:
		using Revision = ark::hal::simd::Sse2;
		using VecFloat4Sse::VecFloat4Sse;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Sse2> with 
	 * VecFloat4Sse2.
	 *
	 * @sa Vec
	 * @sa VecFloat4Sse2
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Sse2>
	{
		typedef VecFloat4Sse2 type;
	};


	//====================================================================
	//  2-D Double-precision Vector
	//====================================================================

	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Floating-point Vector
	 *
	 * @details This class defines the data of a 2-D double-precision 
	 * vector using an SSE2 register format. It conforms to the 
	 * ::ark::hal::Vector concept and is thus compatible with all 
	 * unoptimized functions; however, it is designed to permit 
	 * SSE-optimized functions, and optimized versions of essential 
	 * vector math are defined below.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse2Family
	 */
	class VecDouble2Sse2
	{
		/// Intel intrinsic SSE register format of 2 64-bit floats
		__m128d value_;

	public:
		using Revision = ark::hal::simd::Sse2;
		using Scalar = double;

		/// @name Constructors
		/// @{

		/*****************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/DefaultConstructor.txt
		 */
		VecDouble2Sse2() = default;


		/*****************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		VecDouble2Sse2(Scalar x, Scalar y)
		{
			value_ = _mm_setr_pd(x, y);
		}


		/*****************************************************************
		 * @brief Vector Constructor
		 * @details @includedoc Math/Vector/VectorConstructor.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar>&& SameDimension<VecDouble2Sse2, V>
		VecDouble2Sse2(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
			: VecDouble2Sse2(static_cast<Scalar>(rhs(0)), static_cast<Scalar>(rhs(1)))
		{}


		/*****************************************************************
		 * @brief Raw SIMD Data Constructor
		 * @details @includedoc Math/Vector/SimdConstructor.txt
		 */
		VecDouble2Sse2(__m128d value)
			: value_(value)
		{}

		/// @}

		/*****************************************************************
		 * @brief The number of elements in the vector, its dimension = 4
		 */
		static constexpr size_t Size() noexcept { return 2; }

		/// @name Accessors
		/// @{

		/*****************************************************************
		 * @brief Element Accessor
		 */
		Scalar operator()(size_t index) const noexcept
		{
			__m128d v = SseVal();
			switch(index)
			{
				case 0:
					return _mm_cvtsd_f64(v);

				case 1:
					return _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));

				default:
					// Error condition
					return Scalar(0);
			}
		}


		/*****************************************************************
		 * @brief SSE Data Accessor
		 * @details @includedoc Math/Vector/SimdAccessor.txt
		 */
		__m128d SseVal() const { return value_; }

		/// @}
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 2, Sse2> with 
	 * VecDouble2Sse2.
	 *
	 * @sa Vec
	 * @sa VecDouble2Sse2e
	 */
	template<>
	struct VectorSelector<double, 2, ark::hal::simd::Sse2>
	{
		using type = VecDouble2Sse2;
	};


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Negation
	 * @details @include{doc} Math/Vector/Negation2D.txt
	 *
	 * @supersedes(Vec,operator-(const V& v))
	 * @sa VectorNegation
	 */
	template<VecSse2<double, 2> V>
	inline auto operator-(V v) noexcept -> V
	{
		__m128d result = _mm_sub_pd(_mm_setzero_pd(), v.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Addition
	 * @details @include{doc} Math/Vector/Addition2D.txt
	 *
	 * @supersedes operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 */
	template<VecSse2<double, 2> V>
	inline auto operator+(V lhs, V rhs) noexcept -> V
	{
		__m128d result = _mm_add_pd(lhs.SseVal(), rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Subtraction
	 * @details @include{doc} Math/Vector/Subtraction2D.txt
	 *
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 */
	template<VecSse2<double, 2> V>
	inline auto operator-(V lhs, V rhs) noexcept -> V
	{
		__m128d result = _mm_sub_pd(lhs.SseVal(), rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector-Scalar
	 * Multiplication
	 * @details @include{doc} Math/Vector/VectorScalarMultiplication2D.txt
	 *
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse2<double, 2> V>
	inline auto operator*(V lhs, double rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d result = _mm_mul_pd(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Scalar-Vector
	 * Multiplication
	 * @details @include{doc} Math/Vector/ScalarVectorMultiplication2D.txt
	 *
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse2<double, 2> V>
	inline auto operator*(double lhs, V rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d result = _mm_mul_pd(scalar, rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 2-D Double-precision Vector-Scalar Division
	 * @details @include{doc} Math/Vector/ScalarDivision2D.txt
	 *
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 */
	template<VecSse2<double, 2> V>
	inline auto operator/(V lhs, double rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d result = _mm_div_pd(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Equality
	 * @details @include{doc} Math/Vector/Equality2D.txt
	 *
	 * @sa operator==(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 2> V>
	inline auto operator==(V lhs, V rhs) noexcept -> bool
	{
		__m128d c = _mm_cmpeq_pd(lhs.SseVal(), rhs.SseVal());
		int mask = _mm_movemask_pd(c);
		bool result = mask == 0x3;
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct2D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 2> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d m = _mm_mul_pd(lhs.SseVal(), rhs.SseVal()); // lxrx, lyry
		__m128d s = _mm_shuffle_pd(m, m, _MM_SHUFFLE2(0, 1)); // lyry, lxrx
		__m128d a = _mm_add_pd(m, s); // lxrx+lyry, lxrx+lyry
		double result = _mm_cvtsd_f64(a);
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 2-D Double-precision Vector Cross Product
	 * @details @include{doc} Math/Vector/CrossProduct2D.txt
	 *
	 * @sa Cross(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 2> V>
	inline auto Cross(V lhs, V rhs) noexcept -> double
	{
		__m128d l01 = lhs.SseVal();
		__m128d r01 = rhs.SseVal();

		__m128d r10 = _mm_shuffle_pd(r01, r01, _MM_SHUFFLE2(0, 1)); // r1, r0
		__m128d a01 = _mm_mul_pd(l01, r10); // l0r1, l1r0
		__m128d a10 = _mm_shuffle_pd(a01, a01, _MM_SHUFFLE2(0, 1)); // l1r0, l0r1

		__m128d a = _mm_sub_pd(a01, a10); // l0r1-l1r0, l1r0-l0r1
		double result = _mm_cvtsd_f64(a);
		return result;
	}


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Floating-point Vector
	 *
	 * @details This class defines the data of a 4-D double-precision 
	 * vector using an SSE2 register format. It conforms to the 
	 * ::ark::hal::Vector concept and is thus compatible with all 
	 * unoptimized functions; however, it is designed to permit 
	 * SSE-optimized functions, and optimized versions of essential 
	 * vector math are defined below. The SSE2 specification did not 
	 * extend the size of the registers; they remain 128-bits. As such, 
	 * it requires two data to represent a 4-D vector.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:Sse2Family
	 */
	class VecDouble4Sse2
	{
		/// The double-precision components at indices 0 and 1
		__m128d v01_;
		/// The double-precision components at indices 2 and 3
		__m128d v23_;

	public:
		using Revision = ark::hal::simd::Sse2;
		using Scalar = double;

		/// @name Constructors
		/// @{

		/*****************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/DefaultConstructor.txt
		 */
		VecDouble4Sse2() = default;


		/*****************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		VecDouble4Sse2(Scalar x, Scalar y, Scalar z, Scalar w)
		{
			v01_ = _mm_setr_pd(x, y);
			v23_ = _mm_setr_pd(z, w);
		}

		/*****************************************************************
		 * @brief Vector Constructor
		 * @details @includedoc Math/Vector/VectorConstructor.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar>
			&& SameDimension<Vec, V>
		VecDouble4Sse2(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
			: VecDouble4Sse2(static_cast<Scalar>(rhs(0)), static_cast<Scalar>(rhs(1)), static_cast<Scalar>(rhs(2)), static_cast<Scalar>(rhs(3)))
		{}

			/*****************************************************************
			 * @brief Raw SIMD Data Constructor
			 * 
			 * @param v01 Vector components at indices 0 and 1
			 * @param v23 Vector components at indices 2 and 3
			 * 
			 * @details @includedoc Math/Vector/SimdConstructor.txt
			 */
			VecDouble4Sse2(__m128d v01, __m128d v23)
			: v01_(v01)
			, v23_(v23)
		{}
		/// @}

		/// @name Assignment Functions
		/// @{

		/*****************************************************************
		 * @brief Vector Assignment
		 * @details @includedoc Math/Vector/VectorAssignment.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar>&& SameDimension<VecDouble4Sse2, V>
		VecDouble4Sse2& operator=(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			v01_ = _mm_setr_pd
			(
				static_cast<Scalar>(rhs(0)),
				static_cast<Scalar>(rhs(1))
			);

			v23_ = _mm_setr_pd
			(
				static_cast<Scalar>(rhs(2)),
				static_cast<Scalar>(rhs(3))
			);

			return *this;
		}

		/// @}

		/*****************************************************************
		 * @brief The number of elements in the vector, its dimension = 4
		 */
		static constexpr size_t Size() noexcept { return 4; }


		/// @name Accessors
		/// @{

		/*****************************************************************
		 * @brief Element Accessor
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

		/*****************************************************************
		 * @brief SSE Data Accessor - Elements 0 and 1
		 * @details @includedoc Math/Vector/SimdAccessor.txt
		 */
		__m128d Sse01() const { return v01_; }

		/*****************************************************************
		 * @brief SSE Data Accessor - Elements 2 and 3
		 * @details @includedoc Math/Vector/SimdAccessor.txt
		 */
		__m128d Sse23() const { return v23_; }

		/// @}
	};



	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 4, Sse2> with 
	 * VecDouble4Sse2.
	 *
	 * @sa Vec
	 * @sa VecFloat4Sse
	 */
	template<>
	struct VectorSelector<double, 4, ark::hal::simd::Sse2>
	{
		typedef VecDouble4Sse2 type;
	};


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Negation
	 * @details @include{doc} Math/Vector/Negation4D.txt
	 *
	 * @supersedes(Vec,operator-(const V& v))
	 * @sa VectorNegation
	 */
	template<VecSse2<double, 4> V>
	inline auto operator-(V v) noexcept -> V
	{
		__m128d zero = _mm_setzero_pd();
		__m128d v01 = _mm_sub_pd(zero, v.Sse01());
		__m128d v23 = _mm_sub_pd(zero, v.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Addition
	 * @details @include{doc} Math/Vector/Addition4D.txt
	 *
	 * @supersedes operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 */
	template<VecSse2<double, 4> V>
	inline auto operator+(V lhs, V rhs) noexcept -> V
	{
		__m128d v01 = _mm_add_pd(lhs.Sse01(), rhs.Sse01());
		__m128d v23 = _mm_add_pd(lhs.Sse23(), rhs.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Subtraction
	 * @details @include{doc} Math/Vector/Subtraction4D.txt
	 *
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 */
	template<VecSse2<double, 4> V>
	inline auto operator-(V lhs, V rhs) noexcept -> V
	{
		__m128d v01 = _mm_sub_pd(lhs.Sse01(), rhs.Sse01());
		__m128d v23 = _mm_sub_pd(lhs.Sse23(), rhs.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D double-precision Vector-Scalar
	 * Multiplication
	 * @details @include{doc} Math/Vector/VectorScalarMultiplication4D.txt
	 *
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse2<double, 4> V>
	inline auto operator*(V lhs, double rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d v01 = _mm_mul_pd(lhs.Sse01(), scalar);
		__m128d v23 = _mm_mul_pd(lhs.Sse23(), scalar);
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Scalar-Vector
	 * Multiplication
	 * @details @include{doc} Math/Vector/ScalarVectorMultiplication4D.txt
	 *
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse2<double, 4> V>
	inline auto operator*(double lhs, V rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(lhs);
		__m128d v01 = _mm_mul_pd(scalar, rhs.Sse01());
		__m128d v23 = _mm_mul_pd(scalar, rhs.Sse23());
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector-Scalar Division
	 * @details @include{doc} Math/Vector/ScalarDivision4D.txt
	 *
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 */
	template<VecSse2<double, 4> V>
	inline auto operator/(V lhs, double rhs) noexcept -> V
	{
		__m128d scalar = _mm_set1_pd(rhs);
		__m128d v01 = _mm_div_pd(lhs.Sse01(), scalar);
		__m128d v23 = _mm_div_pd(lhs.Sse23(), scalar);
		return {v01, v23};
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Equality
	 * @details @include{doc} Math/Vector/Equality4D.txt
	 *
	 * @sa operator==(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 4> V>
	inline auto operator==(V lhs, V rhs) noexcept -> bool
	{
		__m128d c01 = _mm_cmpeq_pd(lhs.Sse01(), rhs.Sse01());
		int m01 = _mm_movemask_pd(c01);
		__m128d c23 = _mm_cmpeq_pd(lhs.Sse23(), rhs.Sse23());
		int m23 = _mm_movemask_pd(c23);
		bool mask = (m01 & m23) == 0x3;
		return mask;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m128d v01 = _mm_mul_pd(lhs.Sse01(), rhs.Sse01());
		__m128d v10 = _mm_shuffle_pd(v01, v01, _MM_SHUFFLE2(0, 1));
		__m128d va = _mm_add_pd(v01, v10);

		__m128d v23 = _mm_mul_pd(lhs.Sse23(), rhs.Sse23());
		__m128d v32 = _mm_shuffle_pd(v23, v23, _MM_SHUFFLE2(0, 1));
		__m128d vb = _mm_add_pd(v23, v32);

		__m128d dp = _mm_add_pd(va, vb);
		double result = _mm_cvtsd_f64(dp);
		return result;
	}


	/*********************************************************************
	 * @brief SSE2-optimized 4-D Double-precision Vector Cross Product
	 * @details @include{doc} Math/Vector/CrossProduct4D.txt
	 *
	 * @sa Cross(const V& vl, const V& vr)
	 */
	template<VecSse2<double, 4> V>
	inline auto Cross(V lhs, V rhs) noexcept -> V
	{
		// Gather Data
		__m128d l01 = lhs.Sse01();
		__m128d l23 = lhs.Sse23();
		__m128d r01 = rhs.Sse01();
		__m128d r23 = rhs.Sse23();
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
		__m128d r1z = _mm_shuffle_pd(r01, zzz, _MM_SHUFFLE2(0, 1)); // r1, 0
		__m128d c1a = _mm_mul_pd(l01, r1z); // l0r1, 0

		__m128d l1z = _mm_shuffle_pd(l01, zzz, _MM_SHUFFLE2(0, 1)); // l1, 0
		__m128d c1b = _mm_mul_pd(l1z, r01); // l1r0, 0

		__m128d v23 = _mm_sub_pd(c1a, c1b); // l0r1-l1r0, 0

		// Final result
		return {v01, v23};
	}
}


//========================================================================
#endif
