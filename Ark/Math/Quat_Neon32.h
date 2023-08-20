/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the ARM Neon AArch32.
 *
 * @details This file defines optimizations to the basic Quat class for 
 * use on platforms with ARM CPUs possessing registers and instructions 
 * defined in the Neon AArch32 ISA. This file contains a a class and 
 * supporting non-member functions implementing Quaternion math routines 
 * optimized for the original Neon ISA. The class is then specified as a 
 * specialization of the Quat<S, I> class to be used in client code.
 *
 * @sa ark::math::Quaternion
 * @sa Quat.h
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_NEON32_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_NEON32_H_INCLUDE_GUARD

#include <arm_neon.h>

namespace ark::math
{

	/*********************************************************************
	 * @brief Quaternion with ARM Neon AArch32 Optimization
	 *
	 * @details The concept of quaternion class of a specific scalar type 
	 * optimized for the Neon AArch32 ISA. Functions that implement 
	 * algorithms with Neon AArch32 instructions use this concept to 
	 * define their parameters.
	 *
	 * @sa QuaternionNone
	 * @sa ark::hal::simd::Neon32
	 */
	template<typename Q, typename S>
	concept QuatNeon32 = QuaternionNone<Q, S> &&
		::ark::hal::simd::Neon32Family<typename Q::Revision>;




	/*********************************************************************
	 * @brief Single-precision Floating-point Quaternion Optimized for 
	 * the ARM Neon AArch32 ISA.
	 *
	 * @details This class defines the data type for a Neon 
	 * AArch32-optimized single-precision floating-point quaternion. Data 
	 * access member functions model the Quaternion concept. All 
	 * mathematical operations are defined in non-member functions.
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
	 *     <td>@ref "operator-(QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/ConjugationOperation.f
	 *     <td>@ref "operator*(QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	 *     <td>@ref "Dot(QuatFloatNeon32, QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/InversionOperation.f
	 *     <td>@ref "Inverse(QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/AdditionOperation.f
	 *     <td>@ref "operator+(QuatFloatNeon32, QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/SubtractionOperation.f
	 *     <td>@ref "operator-(QuatFloatNeon32, QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/MultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatNeon32, QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarMultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatNeon32, float)"
	 *   <tr><td>@include{doc} Math/Quaternion/ScalarQuaternionMultiplicationOperation.f
	 *     <td>@ref "operator*(float, QuatFloatNeon32)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarDivisionOperation.f
	 *     <td>@ref "operator/(QuatFloatNeon32, float)"
	 * </table>
	 */
	class QuatFloatNeon32
	{
		/// @name Data
		/// @{

		/// 32-bit Neon-optimized storage of 4 32-bit single-precision floats
		float32x4_t value_;

		/// @}


	public:
		/// @name Configuration Types
		/// @{

		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Neon32;

		/// This specialization is specifically for floats.
		using Scalar = float;

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
		QuatFloatNeon32() noexcept = default;

		/** @brief Component Constructor
		 *  @details Constructor taking the 4 quaternion components
		 *  explicitly as individual parameters.
		 */
		QuatFloatNeon32(Scalar w, Scalar x, Scalar y, Scalar z) noexcept
		{
			float vals[] = { w, x, y, z};
			value_ = vld1q_f32(vals);
		}

		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		QuatFloatNeon32(const Q& rhs) noexcept(IsNoThrowConvertible<Q>)
			: QuatFloatNeon32(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief Neon32 Data Constructor
		 *  @details Constructor to be used by only by Neon32-optimized
		 *  versions of algorithms as it uses a Neon-specific data
		 *  type. Unfortunately, there is no good way to hide it. Do not
		 *  use it in multi-platform code.
		 *  @warning Only use in Neon32-specific algorithm implementations.
		 */
		QuatFloatNeon32(float32x4_t value) noexcept
			: value_(value)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return vgetq_lane_f32(NeonVal(), 0); }
		Scalar x() const noexcept { return vgetq_lane_f32(NeonVal(), 1); }
		Scalar y() const noexcept { return vgetq_lane_f32(NeonVal(), 2); }
		Scalar z() const noexcept { return vgetq_lane_f32(NeonVal(), 3); }

		/** @brief Accessor to Neon32-specific data
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		float32x4_t NeonVal() const noexcept { return value_; }
		/// @}
	};


	/**
	 * @brief Specialize Quat<float, Neon32> with QuatFloatNeon32
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Neon32>
	{
		typedef QuatFloatNeon32 type;
	};


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Negation
	 *
	 * @details Compute a negation of a single-precision floating-point
	 * quaternion using an Neon32-optimized algorithm.
	 * @include{doc} Math/Quaternion/Negation.txt
	 *
	 * @supersedes{QuatFloatNeon32, operator-(const Q&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator-(Q q) noexcept -> Q
	{
		float32x4_t result = vnegq_f32(q.NeonVal());
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Conjugation
	 *
	 * @details Compute a conjugation of a single-precision floating-point
	 * quaternion using a None32-optimized algorithm.
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 *
	 * @supersedes{QuatFloatNeon32, operator*(const Q&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator*(Q q) noexcept -> Q
	{
		float32x4_t value = q.NeonVal();
		float32x4_t negation = -q.NeonVal();
		float32x4_t result = vcopyq_laneq_f32(negation, 0, value, 0);
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Dot Product
	 *
	 * @details Compute the dot product of two single-precision
	 * floating-point quaternions using an algorithm optimized using the 
	 * ARM Neon AArch32 ISA.
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 *
	 * @supersedes{QuatFloatNeon32, Dot(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto Dot(Q lhs, Q rhs) noexcept -> float
	{
		float32x4_t muls = vmulq_f32(lhs.NeonVal(), rhs.NeonVal());
		float32x4_t rev = vrev64q_f32(muls);
		float32x4_t pairs = vaddq_f32(muls, rev);
		float aabb = vgetq_lane_f32(pairs, 0);
		float ccdd = vgetq_lane_f32(pairs, 3);
		float result = aabb + ccdd;
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Addition
	 *
	 * @details Compute the addition of two single-precision floating-
	 * point quaternions using an ARM Neon AArch32-optimized algorithm.
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @supersedes{QuatFloatNeon32, operator+(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator+(Q lhs, Q rhs) noexcept -> Q
	{
		return vaddq_f32(lhs.NeonVal(), rhs.NeonVal());
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Subtraction
	 *
	 * @details Compute the subtraction of one single-precision floating-
	 * point quaternion from another using an ARM Neon AArch32-optimized 
	 * algorithm.
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 *
	 * @supersedes{QuatFloatNeon32, operator-(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator-(Q lhs, Q rhs) noexcept -> Q
	{
		return vsubq_f32(lhs.NeonVal(), rhs.NeonVal());
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion-Scalar 
	 * Multiplication
=	 *
	 * @details Compute the product of a single-precision floating-point
	 * quaternion by a single-precision floating-point scalar value using
	 * an ARM Neon AArch32-optimized algorithm.
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 *
	 * @supersedes{QuatFloatNeon32,operator*(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator*(Q lhs, float rhs) noexcept -> Q
	{
		float32x4_t result = vmulq_n_f32(lhs.NeonVal(), rhs);
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Scalar-Quaternion 
	 * Multiplication
	 *
	 * @details Compute the product of a single-precision floating-point
	 * scalar value and a single-precision floating-point quaternion
	 * value using an ARM Neon AArch32-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 *
	 * @supersedes{QuatFloatNeon32,operator*(typename Q::Scalar\, const Q&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator*(float lhs, Q rhs) noexcept -> Q
	{
		float32x4_t result = vmulq_n_f32(rhs.NeonVal(), lhs);
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Scalar Division
	 *
	 * @details Compute the quotient of a single-precision floating-point
	 * quaternion dividend by a single-precision floating-point scalar
	 * divisor using an ARM Neon32 AArch32-optimized algorithm.
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 *
	 * @supersedes{QuatFloatNeon32,operator/(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	inline auto operator/(Q lhs, float rhs) noexcept -> Q
	{
		float32x4_t scalar = vdupq_n_f32(rhs);
		float32x4_t result = vdivq_f32(lhs.NeonVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief Neon32-optimized Single-precision Quaternion Multiplication
	 *
	 * @details Compute the product of two single-precision floating-point
	 * quaternions using an ARM Neon AArch32-optimized algorithm.
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 *
	 * @supersedes{QuatFloatNeon32,operator*(const QL&\, const QR&)}
	 ********************************************************************/
	template<QuatNeon32<float> Q>
	auto operator*(Q lhs, Q rhs) noexcept -> Q
	{
		static float32_t negations[] =
		{
			 1.0f, -1.0f, -1.0f,
			 1.0f, -1.0f,  1.0f,
			-1.0f,  1.0f, -1.0f,
			-1.0f,  1.0f,  1.0f
		};

		// Gather data
		float32x4_t l = lhs.NeonVal();
		float32x4_t r = rhs.NeonVal();
		float32x4x3_t neg = vld3q_f32(negations);

		// Compute first partial result
		float32x2_t low = vget_low_f32(r);
		float32x4_t r0 = vdupq_lane_f32(low, 0);
		float32x4_t p0 = vmulq_f32(l, r0);

		// Compute 2nd partial result
		float32x4_t zero = vdupq_n_f32(0.0f);
		float32x4_t r1 = vdupq_lane_f32(low, 1);
		float32x4_t m1 = vcmlaq_rot90_f32(zero, r1, l);
		float32x4_t p1 = vfmaq_f32(p0, m1, neg.val[0]);

		// Compute 3rd partial result
		float32x2_t hi = vget_high_f32(r);
		float32x4_t r2 = vdupq_lane_f32(hi, 0);
		float32x4_t ex = vextq_f32(l, l, 2);
		float32x4_t m2 = vmulq_f32(ex, r2);
		float32x4_t p2 = vfmaq_f32(p1, m2, neg.val[1]);

		// Compute 4th partial result
		float32x4_t r3 = vdupq_lane_f32(hi, 1);
		float32x4_t l3 = vrev64q_f32(ex);
		float32x4_t m3 = vmulq_f32(l3, r3);
		float32x4_t result = vfmaq_f32(p2, m3, neg.val[2]);

		return result;
	}
}


//************************************************************************
#endif
