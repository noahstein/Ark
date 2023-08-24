/*************************************************************************
 * @file
 * @brief Quaternion Implementations Optimized for the WebAssembly
 * 128-bit SIMD ISA.
 *
 * @details This file defines optimizations to the basic Quat class for
 * use in browsers that support WebAssembly and its 128-bit SIMD
 * extension. This file contains a a class and supporting non-member
 * functions implementing Quaternion math routines optimized for the
 * original WebAssembly SIMD ISA. The class is then specified as a
 * specialization of the Quat<S, I> class to be used in client code.
 *
 * @sa ark::math::Quaternion
 * @sa Quat.h
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_WASM128_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_WASM128_H_INCLUDE_GUARD

#include <wasm_simd128.h>

namespace ark::math
{

	/*********************************************************************
	 * @brief Quaternion with WebAssembly 128-bit SIMD Optimization
	 *
	 * @details The concept of quaternion class of a specific scalar type
	 * optimized for the WebAssembly 128-bit SIMD ISA. Functions that
	 * implement algorithms with WebAssembly 128-bit SIMD instructions
	 * use this concept to define their parameters.
	 *
	 * @sa QuaternionNone
	 * @sa ark::hal::simd::Neon32
	 */
	template<typename Q, typename S>
	concept QuatWasm128 = QuaternionNone<Q, S> &&
		::ark::hal::simd::Wasm128Family<typename Q::Revision>;




	/*********************************************************************
	 * @brief Single-precision Floating-point Quaternion Optimized for
	 * the WebAssembly 128-bit SIMD ISA.
	 *
	 * @details This class defines the data type for a WebAssembly 128-bit
	 * SIMD single-precision floating-point quaternion. Data access
	 * member functions model the Quaternion concept. All mathematical
	 * operations are defined in non-member functions.
	 *
	 * @note Unlike the optimized data types for SSE, AVX, and Neon; this
	 * class has no associated free function for multiplication. The
	 * initial WebAssembly 128-bit SIMD spec defines a minimal instruction
	 * set designed to work on as many existing hardware implementations
	 * as possible. It continas no lane swizzling instructions.
	 * Consequently, there is real path to optimized multiplication.
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
	 *     <td>@ref "operator-(QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/ConjugationOperation.f
	 *     <td>@ref "operator*(QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/DotProductOperation.f
	 *     <td>@ref "Dot(QuatFloatWasm128, QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/InversionOperation.f
	 *     <td>@ref "Inverse(QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/AdditionOperation.f
	 *     <td>@ref "operator+(QuatFloatWasm128, QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/SubtractionOperation.f
	 *     <td>@ref "operator-(QuatFloatWasm128, QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarMultiplicationOperation.f
	 *     <td>@ref "operator*(QuatFloatWasm128, float)"
	 *   <tr><td>@include{doc} Math/Quaternion/ScalarQuaternionMultiplicationOperation.f
	 *     <td>@ref "operator*(float, QuatFloatWasm128)"
	 *   <tr><td>@include{doc} Math/Quaternion/QuaternionScalarDivisionOperation.f
	 *     <td>@ref "operator/(QuatFloatWasm128, float)"
	 * </table>
	 */
	class QuatFloatWasm128
	{
		/// @name Data
		/// @{

		/// WebAssembly 128-bit SIMD class holding 4 single-precision
		/// floating-point values.
		v128_t value_;

		/// @}


	public:
		/// @name Configuration Types
		/// @{

		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::Wasm128;

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
		QuatFloatWasm128() noexcept = default;

		/** @brief Component Constructor
		 *  @details Constructor taking the 4 quaternion components
		 *  explicitly as individual parameters.
		 */
		QuatFloatWasm128(Scalar w, Scalar x, Scalar y, Scalar z) noexcept
		{
			value_ = wasm_f32x4_make(w, x, y, z);
		}

		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
		QuatFloatWasm128(const Q& rhs) noexcept(IsNoThrowConvertible<Q>)
			: QuatFloatWasm128(static_cast<Scalar>(rhs.w()), static_cast<Scalar>(rhs.x()), static_cast<Scalar>(rhs.y()), static_cast<Scalar>(rhs.z()))
		{}


		/** @brief Neon32 Data Constructor
		 *  @details Constructor to be used by only by Neon32-optimized
		 *  versions of algorithms as it uses a Neon-specific data
		 *  type. Unfortunately, there is no good way to hide it. Do not
		 *  use it in multi-platform code.
		 *  @warning Only use in Neon32-specific algorithm implementations.
		 */
		QuatFloatWasm128(v128_t value) noexcept
			: value_(value)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return wasm_f32x4_extract_lane(WasmVal(), 0); }
		Scalar x() const noexcept { return wasm_f32x4_extract_lane(WasmVal(), 1); }
		Scalar y() const noexcept { return wasm_f32x4_extract_lane(WasmVal(), 2); }
		Scalar z() const noexcept { return wasm_f32x4_extract_lane(WasmVal(), 3); }

		/** @brief Accessor to Neon32-specific data
		 *  @warning Only use in SSE-specific algorithm implementations.
		 */
		v128_t WasmVal() const noexcept { return value_; }
		/// @}
	};


	/**
	 * @brief Specialize Quat<float, Wasm128> with QuatFloatWasm128
	 */
	template<>
	struct QuaternionSelector<float, ark::hal::simd::Wasm128>
	{
		typedef QuatFloatWasm128 type;
	};


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Negation
	 *
	 * @include{doc} Math/Quaternion/Negation.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatWasm128, operator-(const Q&)}
	 ********************************************************************/
	inline auto operator-(QuatFloatWasm128 q) noexcept -> QuatFloatWasm128
	{
		v128_t result = wasm_f32x4_neg(q.WasmVal());
		return result;
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Conjugation
	 *
	 * @include{doc} Math/Quaternion/Conjugation.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatWasm128, operator*(const Q&)}
	 ********************************************************************/
	inline auto operator*(QuatFloatWasm128 q) noexcept -> QuatFloatWasm128
	{
		v128_t value = q.WasmVal();
		v128_t negation = (-q).WasmVal();
		float w = wasm_f32x4_extract_lane(value, 0);
		v128_t result = wasm_f32x4_replace_lane(negation, 0, w);
		return result;
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Dot Product
	 *
	 * @include{doc} Math/Quaternion/DotProduct.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatWasm128, Dot(const QL&\, const QR&)}
	 ********************************************************************/
	inline auto Dot(QuatFloatWasm128 lhs, QuatFloatWasm128 rhs) noexcept -> float
	{
		v128_t mul = wasm_f32x4_mul(lhs.WasmVal(), rhs.WasmVal());
		float w = wasm_f32x4_extract_lane(mul, 0);
		float x = wasm_f32x4_extract_lane(mul, 1);
		float y = wasm_f32x4_extract_lane(mul, 2);
		float z = wasm_f32x4_extract_lane(mul, 3);
		float result = w + x + y + z;
		return result;
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Addition
	 *
	 * @include{doc} Math/Quaternion/Addition.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatNeon32, operator+(const QL&\, const QR&)}
	 ********************************************************************/
	inline auto operator+(QuatFloatWasm128 lhs, QuatFloatWasm128 rhs) noexcept -> QuatFloatWasm128
	{
		return wasm_f32x4_add(lhs.WasmVal(), rhs.WasmVal());
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Subtraction
	 *
	 * @details Compute the subtraction of one single-precision floating-
	 * point quaternion from another using an ARM Neon AArch32-optimized
	 * algorithm.
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatNeon32, operator-(const QL&\, const QR&)}
	 ********************************************************************/
	inline auto operator-(QuatFloatWasm128 lhs, QuatFloatWasm128 rhs) noexcept -> QuatFloatWasm128
	{
		return wasm_f32x4_sub(lhs.WasmVal(), rhs.WasmVal());
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion-Scalar Multiplication
	 *
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatNeon32,operator*(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	inline auto operator*(QuatFloatWasm128 lhs, float rhs) noexcept -> QuatFloatWasm128
	{
		v128_t scalar = wasm_f32x4_splat(rhs);
		v128_t result = wasm_f32x4_mul(lhs.WasmVal(), scalar);
		return result;
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Scalar-Quaternion Multiplication
	 *
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatNeon32,operator*(typename Q::Scalar\, const Q&)}
	 ********************************************************************/
	inline auto operator*(float lhs, QuatFloatWasm128 rhs) noexcept -> QuatFloatWasm128
	{
		v128_t scalar = wasm_f32x4_splat(lhs);
		v128_t result = wasm_f32x4_mul(scalar, rhs.WasmVal());
		return result;
	}


	/*********************************************************************
	 * @brief WebAssembly 128-bit SIMD-optimized Single-precision
	 * Quaternion Scalar Division
	 *
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * @include{doc} Emscripten-concpet-aimbiguous-overload.txt
	 * @supersedes{QuatFloatNeon32,operator/(const Q&\, typename Q::Scalar)}
	 ********************************************************************/
	inline auto operator/(QuatFloatWasm128 lhs, float rhs) noexcept -> QuatFloatWasm128
	{
		v128_t scalar = wasm_f32x4_splat(rhs);
		v128_t result = wasm_f32x4_div(lhs.WasmVal(), scalar);
		return result;
	}
}


//************************************************************************
#endif
