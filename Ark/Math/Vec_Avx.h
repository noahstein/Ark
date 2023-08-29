/*************************************************************************
 * @file
 * @brief AVX-optimized Vec Implementation
 *
 * @details This file defines optimizations to the Vec type family
 * utilizing the AVX SIMD ISA. AVX is an extension to SSE4. It introduces 
 * a new 256-bit type, providing for 4-element double-precision values; 
 * therefore, this version introduces a new type:
 * 
 * - VecDouble4AVX: 4-D double-precision vector (Vec<double, 4, Avx>)
 *
 * @sa Vec.h
 * @se Vec_Sse4.h
 * @sa ::ark::hal::simd::Avx
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
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief AVX-optimized Vector Parameter Concept
	 *
	 * @tparam V The type of vector client code is passing as an argument
	 * @tparam S The type of scalar the function requires
	 * @tparam N The dimension of the vector the function requires
	 *
	 * @details Concept for declaring parameters of AVX-optimized
	 * functions.It ensures the parameter is of a type optimized for the
	 * AVX SIMD ISA but not all of the earlier SSE specifications.
	 * @includedoc Math/Vector/ParameterConcept.txt
	 *
	 * @sa ark::math::Vec
	 * @sa ark::hal::simd::AVxFamily
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecAvx = VecSse4<V, S, N> &&
		::ark::hal::simd::AvxFamily<typename V::Revision>;


	//====================================================================
	//  ISA Register Set Promotion
	//====================================================================

	/*********************************************************************
	 * @brief AVX-optimized 4-D Single-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE4 to AVX do not result in a
	 * new register specification applicable to further 4-D
	 * single-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:AvxFamily
	 */
	class VecFloat4Avx : public VecFloat4Sse4
	{
	public:
		using Revision = ark::hal::simd::Avx;
		using VecFloat4Sse4::VecFloat4Sse4;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<float, 4, Avx> with
	 * VecFloat4Avx.
	 *
	 * @sa Vec
	 * @sa VecFloat4Avx
	 */
	template<>
	struct VectorSelector<float, 4, ark::hal::simd::Avx>
	{
		using type = VecFloat4Avx;
	};


	/*********************************************************************
	 * @brief AVX-optimized 2-D Double-precision Floating-point Vector
	 *
	 * @details The changes changes from SSE4 to AVX do not result in a
	 * new register specification applicable to further 2-D
	 * double-precision floating-point vectors.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math:AvxFamily
	 */
	class VecDouble2Avx : public VecDouble2Sse4
	{
	public:
		using Revision = ark::hal::simd::Avx;
		using VecDouble2Sse4::VecDouble2Sse4;
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 2, Avx> with
	 * VecDouble2Avx.
	 *
	 * @sa Vec
	 * @sa VecDouble2Avx
	 */
	template<>
	struct VectorSelector<double, 2, ark::hal::simd::Avx>
	{
		typedef VecDouble2Avx type;
	};


	//====================================================================
	//  4-D Vec Double Specialization
	//====================================================================

	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Floating-point Vector
	 *
	 * @details This class defines the data of a 4-D double-precision
	 * vector using an AVX register format. The new AVX ISA extends 
	 * register to 256 bits, enabling a single-register 4-D 
	 * double-precision vector. Consequently, all prior SSEx-optimized 
	 * functions will not work with the new data class. It conforms to 
	 * the ::ark::hal::Vector concept and is thus compatible with all
	 * unoptimized functions; however, it is designed to permit
	 * AVX-optimized functions, and optimized versions of essential
	 * vector math are defined below.
	 *
	 * @sa Vec
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math::AvxFamily
	 */
	class VecDouble4Avx
	{
		/// AVX-optimized storage of 4 64-bit single-precision floats
		__m256d value_;

	public:
		using Revision = ark::hal::simd::Avx;
		using Scalar = double;

		/// @name Constructors
		/// @{

		/*****************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/DefaultConstructor.txt
		 */
		VecDouble4Avx() = default;


		/*****************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		VecDouble4Avx(Scalar x, Scalar y, Scalar z, Scalar w)
		{
			value_ = _mm256_setr_pd(x, y, z, w);
		}

		/*****************************************************************
		 * @brief Vector Constructor
		 * @details @includedoc Math/Vector/VectorConstructor.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && SameDimension<VecDouble4Avx, V>
		VecDouble4Avx(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
			: VecDouble4Avx(static_cast<Scalar>(rhs(0)), static_cast<Scalar>(rhs(1)), static_cast<Scalar>(rhs(2)), static_cast<Scalar>(rhs(3)))
		{}

		/*****************************************************************
		 * @brief Raw SIMD Data Constructor
		 * @details @includedoc Math/Vector/SimdConstructor.txt
		 */
		VecDouble4Avx(__m256d value)
			: value_(value)
		{}

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

		/*****************************************************************
		 * @brief AVX Data Accessor
		 * @details @includedoc Math/Vector/SimdAccessor.txt
		 */
		__m256d AvxVal() const { return value_; }

		/// @}
	};


	/*********************************************************************
	 * @brief Specialize VectorSelector<double, 4, Avx> with 
	 * VecDouble4Avx.
	 *
	 * @sa Vec
	 * @sa VecFloatAvx
	 */
	template<>
	struct VectorSelector<double, 4, ark::hal::simd::Avx>
	{
		using type = VecDouble4Avx;
	};


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Negation
	 * @details @include{doc} Math/Vector/Negation4D.txt
	 *
	 * @supersedes(Vec,operator-(const V& v))
	 * @sa VectorNegation
	 */
	template<VecAvx<double, 4> V>
	inline auto operator-(V v) noexcept -> V
	{
		__m256d result = _mm256_sub_pd(_mm256_setzero_pd(), v.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Addition
	 * @details @include{doc} Math/Vector/Addition4D.txt
	 *
	 * @supersedes operator+(const V& vl, const V& vr)
	 * @sa VectorAddition
	 */
	template<VecAvx<double, 4> V>
	inline auto operator+(V lhs, V rhs) noexcept -> V
	{
		__m256d result = _mm256_add_pd(lhs.AvxVal(), rhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Subtraction
	 * @details @include{doc} Math/Vector/Subtraction4D.txt
	 *
	 * @sa operator-(const V& vl, const V& vr)
	 * @sa VectorSubtraction
	 */
	template<VecAvx<double, 4> V>
	inline auto operator-(V lhs, V rhs) noexcept -> V
	{
		__m256d result = _mm256_sub_pd(lhs.AvxVal(), rhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D double-precision Vector-Scalar
	 * Multiplication
	 * @details @include{doc} Math/Vector/VectorScalarMultiplication4D.txt
	 *
	 * @sa operator*(const V& v, const S& s)
	 * @sa VectorScalarMultiplication
	 */
	template<VecAvx<double, 4> V>
	inline auto operator*(V lhs, double rhs) noexcept -> V
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_mul_pd(lhs.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Scalar-Vector
	 * Multiplication
	 * @details @include{doc} Math/Vector/ScalarVectorMultiplication4D.txt
	 *
	 * @sa operator*(const S& s, const V& v)
	 * @sa VectorScalarMultiplication
	 */
	template<VecAvx<double, 4> V>
	inline auto operator*(double lhs, V rhs) noexcept -> V
	{
		__m256d scalar = _mm256_set1_pd(lhs);
		__m256d result = _mm256_mul_pd(scalar, rhs.AvxVal());
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector-Scalar Division
	 * @details @include{doc} Math/Vector/ScalarDivision4D.txt
	 *
	 * @sa operator/(const V& v, const S& s)
	 * @sa VectorScalarDivision
	 */
	template<VecAvx<double, 4> V>
	inline auto operator/(V lhs, double rhs) noexcept -> V
	{
		__m256d scalar = _mm256_set1_pd(rhs);
		__m256d result = _mm256_div_pd(lhs.AvxVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Equality
	 * @details @include{doc} Math/Vector/Equality4D.txt
	 *
	 * @sa operator==(const V& vl, const V& vr)
	 */
	template<VecAvx<double, 4> V>
	inline auto operator==(V lhs, V rhs) noexcept -> bool
	{
		__m256d c = _mm256_cmp_pd(lhs.AvxVal(), rhs.AvxVal(), _CMP_EQ_OQ);
		int mask = _mm256_movemask_pd(c);
		bool result = mask == 0xf;
		return result;
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Dot Product
	 * @details @include{doc} Math/Vector/DotProduct4D.txt
	 *
	 * @sa Dot(const V& vl, const V& vr)
	 */
	template<VecAvx<double, 4> V>
	inline auto Dot(V lhs, V rhs) noexcept -> double
	{
		__m256d x_y_z_w = _mm256_mul_pd(lhs.AvxVal(), rhs.AvxVal());
		__m256d xy_zw = _mm256_hadd_pd(x_y_z_w, x_y_z_w);
		__m256d zw_xy = _mm256_permute2f128_pd(xy_zw, xy_zw, 5);
		__m256d xyzw = _mm256_add_pd(xy_zw, zw_xy);
		return _mm256_cvtsd_f64(xyzw);
	}


	/*********************************************************************
	 * @brief AVX-optimized 4-D Double-precision Vector Cross Product
	 * @details @include{doc} Math/Vector/CrossProduct4D.txt
	 *
	 * @sa Cross(const V& vl, const V& vr)
	 */
	template<VecAvx<double, 4> V>
	inline auto Cross(V lhs, V rhs) noexcept -> V
	{
		__m256d l = lhs.AvxVal();
		__m256d r = rhs.AvxVal();

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
