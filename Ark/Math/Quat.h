/*************************************************************************
 * @file
 * @brief Basic, platform-independent implementation of a quaternion.
 * 
 * @details This file defines the basic quaternion class intended for 
 * general use in application code.
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUAT_H_INCLUDE_GUARD)
#define ARK_MATH_QUAT_H_INCLUDE_GUARD


//************************************************************************
// Dependencies
//************************************************************************
#include "Quaternion.h"
#include "Ark/Hal/Simd/Simd.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief Quaternion with no SIMD Optimization
	 * 
	 * @tparam Q The quaternion class used with this concept
	 * @tparam S The type of the scalar desire in the usage.
	 * 
	 * @details The concept of quaternion with no SIMD optimization is 
	 * essentially the concept of a quaternion of a given scalar type. 
	 * This concept ensures the template type meets the Quaternion concept 
	 * and checks to see the Scalar is of a given Type. This concept is 
	 * designed to be used to perform the basic quaternion concept checks 
	 * all SIMD-optimized Quaternion classes need to do.
	 * 
	 * @sa Quaternion
	 * @sa SimdConceptHierarchy
	 */
	template<typename Q, typename S>
	concept QuaternionNone = Quaternion<Q> &&
		std::same_as<typename Q::Scalar, S>;


	/*********************************************************************
	 * @brief Basic Quaternion Class
	 * 
	 * @tparam S The type of the scalar components.
	 * 
	 * @details A basic quaternion class definition templated on the 
	 * scalar type. It is a simple, dense storage of 4 scalars. It will 
	 * work on any platform although it will only use specialized SIMD 
	 * hardware if the compiler can optimize for it. This class is the 
	 * default one chosen by Quat.
	 * 
	 * There are no accompanying operations. All operations are used by 
	 * the generic template expression trees. There may be a tendency for 
	 * the code to bloat as the trees tend to inline most every operation; 
	 * however, the optimizer has maximum potential for optimizing 
	 * algorithms as the alternative is to transfer between memory with 
	 * function calls.
	 *
	 * @subsubsection Concepts Concepts
	 * @par
	 * @ref "ark::math::Quaternion" "Quaternion"
	 ********************************************************************/
	template<typename S>
	class QuatBasic
	{
	public:
		/// @name Configuration Types
		/// @{

		/// Tag specifying the SIMD revision ID
		using Revision = ark::hal::simd::None;

		/** The numeric type of the components is defined by the first
		 *  class template parameter.
		 */
		using Scalar = S;

		/// @}

		/// @name noexcept Tests
		/// @{

		/** @brief Are instances of Scalar nothrow default constructible?
		 *  @details If instances of Scalar are nothrow default 
		 *  constructible then this QuatBasic class is also nothrow 
		 *  default constructible as it is composed of only 4 Scalar 
		 *  instances.
		 */
		constexpr static bool IsNoThrowDefaultConstructible = std::is_nothrow_default_constructible_v<Scalar>;

		/** @brief Are instances of Scalar nothrow copy constructible?
		 *  @details If instances of Scalar are nothrow copy 
		 *  constructible then this QuatBasic class is also nothrow copy 
		 *  constructible as it is only composed of only 4 Scalar 
		 *  instances.
		 */
		constexpr static bool IsNoThrowCopy = std::is_nothrow_copy_constructible_v<Scalar>;

		/** @brief Can a conversion from Q class's Scalar be nothrow converted to Scalar?
		 *  @tparam Q The type of the quaternion being copied from
		 *  @details If another quaternion's Scalar values may be nothrow
		 *  converted to this QuaternBasic's Scalar type then the copy 
		 *  is also nothrow as it is only composed of 4 Scalar instances.
		 */
		template<typename Q>
		constexpr static bool IsNoThrowConvertible = std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>;

		/// @}

	private:
		Scalar w_, x_, y_, z_;

	public:
		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details The default constructor leaves storage 
		 *  uninitialized, just like the behavior of built-in types.
		 */			
		QuatBasic() noexcept(IsNoThrowDefaultConstructible) = default;

		/** @brief Component-wise Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individual parameters.
		 */
		QuatBasic(Scalar w, Scalar x, Scalar y, Scalar z) noexcept(IsNoThrowCopy)
			: w_(w), x_(x), y_(y), z_(z)
		{}

		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		QuatBasic(const Q& rhs) noexcept(IsNoThrowConvertible<Q>)
			: QuatBasic(Scalar{rhs.w()}, Scalar{rhs.x()}, Scalar{rhs.y()}, Scalar{rhs.z()})
		{}
		/// @}

		/** @brief Quaternion Concept Assignment Operator
		 *  @details Assign the quaternion's value from an instance of 
		 *  any class meeting the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		QuatBasic& operator=(const Q& rhs) noexcept(IsNoThrowConvertible<Q>)
		{
			w_ = Scalar{rhs.w()};
			x_ = Scalar{rhs.x()};
			y_ = Scalar{rhs.y()};
			z_ = Scalar{rhs.z()};

			return *this;
		}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept(IsNoThrowCopy) { return w_; }
		Scalar x() const noexcept(IsNoThrowCopy) { return x_; }
		Scalar y() const noexcept(IsNoThrowCopy) { return y_; }
		Scalar z() const noexcept(IsNoThrowCopy) { return z_; }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Selector Template
	 *
	 * @tparam S The type of the scalar components.
	 * @tparam I The SIMD architecture
	 *
	 * @details The template defines a single typedef named "type" to 
	 * define which quaternion class should be used given the scalar type 
	 * S and SIMD architecture I. WHen a SIMD ISA provides facilities to 
	 * optimize algorithms, create any new implementations for that ISA, 
	 * and then define a specialization of this template to enable Quat 
	 * to automatically select it when that implementation is desired.
	 ********************************************************************/
	template<typename S, typename I = ark::hal::simd::HAL_SIMD>
	struct QuaternionSelector
	{
		typedef QuatBasic<S> type;
	};


	/*********************************************************************
	 * @brief Standard Dense Quaternion
	 *
	 * @tparam S The type of the scalar components.
	 * @tparam I The SIMD architecture
	 *
	 * @details Programmers who desire to use a "normal" quaternion should 
	 * use this type alias. It is designed to automatically pick the best 
	 * class based on the scalar type and the SIMD ISA, falling back to 
	 * the default implementation QuatBasic when there is no better SIMD 
	 * ISA. To do this, it uses QuaternionSelector to select the optimal 
	 * type based on the template parameters.
	 ********************************************************************/
	template<typename S, typename I = ark::hal::simd::HAL_SIMD>
	using Quat = QuaternionSelector<S, I>::type;
}


//========================================================================
//	Platform-optimized Specializations
//========================================================================
#if __has_include(INCLUDE_SIMD(Quat))
#include INCLUDE_SIMD(Quat)
#endif


//========================================================================
#endif
