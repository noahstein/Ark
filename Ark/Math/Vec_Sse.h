/*************************************************************************
 * @file
 * @brief SSE-optimized Vec Implementation
 * 
 * @details This file defines optimizations to the Vec type family 
 * utilizing the SSE SIMD ISA. This includes not just the class 
 * implementing the optimized format and the supporting non-member 
 * functions but also includes any supporting infrastructure necessary to 
 * enable the optimizations. The ISA's register format and instructions 
 * permit the following optimization class:
 * 
 * - VecFloat4Sse: 4-D single-precision vector (Vec<float, 4, Sse>)
 * 
 * @sa Vec.h
 * @sa ::ark::hal::simd::Sse
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
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
	//====================================================================
	// Concepts
	//====================================================================

	/*********************************************************************
	 * @brief SSE-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 * 
	 * @details Concept for declaring parameters of SSE-optimized 
	 * functions.It ensures the parameter is of a type optimized for the 
	 * SSE SIMD ISA.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 * 
	 * @sa ark::math::VecNone
	 * @sa ark::hal::simd::SseFamily
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecSse = 
		VecNone<V, S, N> &&
		::ark::hal::simd::SseFamily<typename V::Revision>;


	//====================================================================
	//  4-D Single-precision Vector
	//====================================================================

	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Floating-point Vector
	 * 
	 * @details This class defines the data of a 4-D vector using an SSE 
	 * data type. It conforms to the ::ark::hal::Vector concept and is 
	 * thus compatible with all unoptimized functions; however, it is 
	 * designed to permit SSE-optimized functions, and optimized versions 
	 * of essential vector math are defined below.
	 * 
	 * @sa Vec
	 * @sa ::ark::hal::Vector
	 * @sa ::ark::hal::simd::Sse
	 */
	class VecFloat4Sse
	{
		/// Intel intrinsic SSE register format of 4 32-bit floats
		__m128 value_;

	public:
		using Revision = ark::hal::simd::Sse;
		using Scalar = float;

		/// @name Constructors
		/// @{

		/*****************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/DefaultConstructor.txt
		 */
		VecFloat4Sse() = default;


		/*****************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		VecFloat4Sse(Scalar x, Scalar y, Scalar z, Scalar w)
		{
			value_ = _mm_setr_ps(x, y, z, w);
		}


		/*****************************************************************
		 * @brief Vector Constructor
		 * @details @includedoc Math/Vector/VectorConstructor.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && SameDimension<Vec, V>
		VecFloat4Sse(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
			: Vec(static_cast<Scalar>(rhs(0)), static_cast<Scalar>(rhs(1)), static_cast<Scalar>(rhs(2)), static_cast<Scalar>(rhs(3)))
		{}


		/*****************************************************************
		 * @brief Raw SIMD Data Constructor
		 * @details @includedoc Math/Vector/SimdConstructor.txt
		 */
		VecFloat4Sse(__m128 value)
			: value_(value)
		{}

		/// @}


		/// @name Assignment Functions
		/// @{

		/*****************************************************************
		 * @brief Vector Assignment
		 * @details @includedoc Math/Vector/VectorAssignment.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && SameDimension<Vec, V>
		VecFloat4Sse & operator=(const V & rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			value_ = _mm_setr_ps
			(
				static_cast<Scalar>(rhs(0)),
				static_cast<Scalar>(rhs(1)),
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


		/*****************************************************************
		 * @brief SSE Data Accessor
		 * @details @includedoc Math/Vector/SimdAccessor.txt
		 */
		__m128 SseVal() const noexcept { return value_; }

		/// @}

	private:
		 /****************************************************************
		  * @brief Get the Nth component in the SSE vector.
		  * @tparam I The index of the component to retrieve.
		  * @param v The SSE data of the vector
		  * @return float The value of the individual component
		  */
		template<int I> inline static auto GetNth(__m128 v) noexcept -> float
		{
			return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(I, I, I, I)));
		};
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Sse> with VecFloat4Sse.
	 *
	 * @sa Vec
	 * @sa VecFloat4Sse
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Sse>
	{
		using type = VecFloat4Sse;
	};


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Negation
	 * @details @include{doc} Math/Vector/Negation4D.txt
	 * 
	 * @supersedes(Vec,operator-(const V& v))
	 * @sa VectorNegation
	 */
	template<VecSse<float, 4> V>
	inline auto operator-(V v) noexcept -> V
	{
		__m128 result = _mm_sub_ps(_mm_setzero_ps(), v.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Addition
	 * @details @include{doc} Math/Vector/Addition4D.txt
	 * 
	 * @supersedes operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 */
	template<VecSse<float, 4> V>
	inline auto operator+(V lhs, V rhs) noexcept -> V
	{
		__m128 result = _mm_add_ps(lhs.SseVal(), rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Subtraction
	 * @details @include{doc} Math/Vector/Subtraction4D.txt
	 * 
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 */
	template<VecSse<float, 4> V>
	inline auto operator-(V lhs, V rhs) noexcept -> V
	{
		__m128 result = _mm_sub_ps(lhs.SseVal(), rhs.SseVal());
		return result;
	}



	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector-Scalar 
	 * Multiplication
	 * @details @include{doc} Math/Vector/VectorScalarMultiplication4D.txt
	 * 
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse<float, 4> V>
	inline auto operator*(V lhs, float rhs) noexcept -> V
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_mul_ps(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Scalar-Vector 
	 * Multiplication
	 * @details @include{doc} Math/Vector/ScalarVectorMultiplication4D.txt
	 * 
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 */
	template<VecSse<float, 4> V>
	inline auto operator*(float lhs, V rhs) noexcept -> V
	{
		__m128 scalar = _mm_set1_ps(lhs);
		__m128 result = _mm_mul_ps(scalar, rhs.SseVal());
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector-Scalar Division
	 * @details @include{doc} Math/Vector/ScalarDivision4D.txt
	 * 
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 */
	template<VecSse<float, 4> V>
	inline auto operator/(V lhs, float rhs) noexcept -> V
	{
		__m128 scalar = _mm_set1_ps(rhs);
		__m128 result = _mm_div_ps(lhs.SseVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Equality
	 * @details @include{doc} Math/Vector/Equality4D.txt
	 * 
	 * @sa operator==(const V& vl, const V& vr)
	 */
	template<VecSse<float, 4> V>
	inline auto operator==(V lhs, V rhs) noexcept -> bool
	{
		__m128 c = _mm_cmpeq_ps(lhs.SseVal(), rhs.SseVal());
		int mask = _mm_movemask_ps(c);
		bool result = mask == 0xf;
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 * 
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecSse<float, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> float
	{
		__m128 m = _mm_mul_ps(lhs.SseVal(), rhs.SseVal()); // lxrx, lyry, lzrz, lwrw
		__m128 s = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 1, 2, 3)); // lwrw, lzrz, lyry, lxrx
		__m128 m2 = _mm_add_ps(m, s); // lxrx+lwrw, lyry+lzrz, lyry+lzrz, lxrx+lwrw
		__m128 s2 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1, 0, 3, 2)); // lyry+lzrz, lxrx+lwrw, lyry+lzrz, lxrx+lwrw
		__m128 m3 = _mm_add_ps(m2, s2); // lxrx+lyry+lzrz+lwrw, ...
		float result = _mm_cvtss_f32(m3);
		return result;
	}


	/*********************************************************************
	 * @brief SSE-optimized 4-D Single-precision Vector Cross Product
	 * @details @include{doc} Math/Vector/CrossProduct4D.txt
	 * 
	 * @sa Cross(const V& vl, const V& vr)
	 */
	template<VecSse<float, 4> V>
	inline auto Cross(V lhs, V rhs) noexcept -> V
	{
		__m128 l = lhs.SseVal();
		__m128 r = rhs.SseVal();

		__m128 rs = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 0, 2, 1)); // ry, rz, rx, rw
		__m128 ls = _mm_shuffle_ps(l, l, _MM_SHUFFLE(3, 0, 2, 1)); // ly, lz, lx, lw
		__m128 lr = _mm_mul_ps(l, rs); // lx*ry, ly*rz, lz*rx, lw*rw
		__m128 rl = _mm_mul_ps(r, ls); // ly*rx, lz*ry, lx*rz, lw*rw
		__m128 a = _mm_sub_ps(lr, rl); // lx*ry-ly*rx, ly*rz-lz*ry, lz*rx-lx*rz, 0
		__m128 result = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1 )); // ly*rz-lz*ry, lz*rx-lx*rz, lx*ry-ly*rx, 0
		return result;
	}
}


//========================================================================
#endif
