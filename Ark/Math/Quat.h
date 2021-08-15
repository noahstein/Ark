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
	 * @tparam I An ISA tag permitting speciaiization of the Quat's data  
	 * layout to optimized for a specific SIMD architectur.e
	 * 
	 * @details The standard small, dense quaternion class. This is the 
	 * go-to class to commonlhy use. It is intended to be the default 
	 * choice of classes when using quaternions in code. It is a very
	 * simple class as all the heavy lifting is done wiether in the
	 * template expression tree classes or in platform-specific 
	 * specializations. This class has only has three simple 
	 * responsibilities it must fulfill as defined in the Quaternion 
	 * concept:
	 * 
	 * -# Define the scalar type of the stored component numbers.
	 * -# Define the storage of the 4 components.
	 * -# Define accessors to the w, x, y, and z components.
	 * 
	 * With these requirements fulfilled, this class automatically 
	 * participates in all functions and classes accepting Quaternion 
	 * concept parameters. That means, although this class defines no 
	 * operations beyond accessors, it will participate in all quaternion 
	 * algebraic functions and algorithms implemented to the concept.
	 * 
	 * @sa @ref ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	template<typename S, typename I = ark::hal::simd::HAL_SIMD>
	class Quat
	{
	public:
		/** The numeric type of the components is defined by the first 
		 *  class template parameter.
		 */
		using Scalar = S;

	private:
		Scalar w_, x_, y_, z_;

	public:
		/// @name Constructors
		/// @{

		/** @brief Default Constructor
		 *  @details The default constructor leaves storage 
		 *  uninitialized, just like the behavior of built-in types.
		 */			
		Quat() = default;

		/** @brief Compopnent-wise Constructor
		 *  @details Constructor taking the 4 quaternion components 
		 *  explicitly as separate, individaul parameters.
		 */
		Quat(Scalar w, Scalar x, Scalar y, Scalar z) noexcept(std::is_nothrow_copy_constructible_v<Scalar>)
			: w_(w), x_(x), y_(y), z_(z)
		{}

		/** @brief Quaternion Concept Constructor
		 *  @details Constructor from any type that is compatible with 
		 *  the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		Quat(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
			: Quat(Scalar{rhs.w()}, Scalar{rhs.x()}, Scalar{rhs.y()}, Scalar{rhs.z()})
		{}
		/// @}

		/** @brief Quaternion Concept Assignment Operator
		 *  @details Assign the quaternion's value from an instance of 
		 *  any clas meeting the Quaternion concept.
		 */
		template<Quaternion Q>
			requires std::convertible_to<typename Q::Scalar, Scalar>
		Quat& operator=(const Q& rhs) noexcept(std::is_nothrow_convertible_v<typename Q::Scalar, Scalar>)
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
}


//========================================================================
//	Platform-optimized Specializations
//========================================================================
#if __has_include(INCLUDE_SIMD(Quat))
#include INCLUDE_SIMD(Quat)
#endif


//========================================================================
#endif
