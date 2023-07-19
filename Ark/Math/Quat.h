/*************************************************************************
 * @file
 * @brief Basic, platform-independent implementation of a quaternion.
 * 
 * @details This file defines the basic quaternion class intended for 
 * general use in application code.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
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
	 * @brief Basic Quaternion Class
	 * 
	 * @tparam S The type of the scalar components.
	 * 
	 * @details A basic quaternion class definition templatized on the 
	 * scalar type. It is a simple, dense stoarage of 4 scalars. It will 
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
		/** The numeric type of the components is defined by the first 
		 *  class template parameter.
		 */
		using Scalar = S;

		/// Tag for revision this implementation's generation in the SIMD family.
		using Revision = ark::hal::simd::None;

	private:
		Scalar w_, x_, y_, z_;

	public:
		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details The default constructor leaves storage 
		 *  uninitialized, just like the behavior of built-in types.
		 */			
		QuatBasic() = default;

		/** @brief Compopnent-wise Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individaul parameters.
		 */
		QuatBasic(Scalar w, Scalar x, Scalar y, Scalar z) noexcept(std::is_nothrow_copy_constructible_v<Scalar>)
			: w_(w), x_(x), y_(y), z_(z)
		{}

		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		QuatBasic(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
			: QuatBasic(Scalar{rhs.w()}, Scalar{rhs.x()}, Scalar{rhs.y()}, Scalar{rhs.z()})
		{}
		/// @}

		/** @brief Quaternion Concept Assignment Operator
		 *  @details Assign the quaternion's value from an instance of 
		 *  any clas meeting the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		QuatBasic& operator=(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
		{
			w_ = Scalar{rhs.w()};
			x_ = Scalar{rhs.x()};
			y_ = Scalar{rhs.y()};
			z_ = Scalar{rhs.z()};

			return *this;
		}

		/// @name Accessors
		/// @{
		Scalar w() const { return w_; }
		Scalar x() const { return x_; }
		Scalar y() const { return y_; }
		Scalar z() const { return z_; }
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
