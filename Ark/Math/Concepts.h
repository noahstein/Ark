/*************************************************************************
 * @file
 * @brief General Mathematical Concepts.
 * 
 * @details The Ark library's mathematics module relies upon C++ 20's 
 * constraints and concepts for its design and implementation. This file 
 * contains those concepts useful for general algorithmic implementation.
 * 
 * @sa https://en.cppreference.com/w/cpp/language/constraints
 * @sa https://en.cppreference.com/w/cpp/concepts
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_CONCEPTS_H_INCLUDE_GUARD)
#define ARK_MATH_CONCEPTS_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <type_traits>


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	/*********************************************************************
	 * @brief A type that implements arithmetic operations.
	 * 
	 * @details This concept models a type that has standard arithmetic 
	 * operations defined on it. Mathematically, the standard arithmetic 
	 * operatoins are semi-groups. As such, at a minimum, they meet the 
	 * following requirements for each operation \f$\otimes\f$ given two 
	 * arguments of the same `Arithmetic` type \f$\textrm{T}\f$:
	 * 
	 * - Closure: \f$a \otimes b \in \textrm{T}\f$
	 * - Associativity: \f$(a \otimes b) \otimes c = 
	 *                  a \otimes (b \otimes c)\f$
	 * - Identity: \f$ \exists I \in T: 
	 *             a \otimes I = I \otimes a = a\f$
	 * - Inverse: \f$a \otimes a^{-1} = a^{-1} \otimes a = 1\f$
	 * 
	 * @note The current implementation is based on the standard's 
	 * `is_arithmetic_v<>` template. If custom types cannot participate in 
	 * its resolution then it will become necessary to channge this 
	 * implementation to something else that can support custom types.
	 ********************************************************************/
	template<typename T>
	concept Arithmetic = std::is_arithmetic_v<T>;


	/*********************************************************************
	 * @brief Two types that may participate together in arithmetic.
	 * 
	 * @details This concept models two types used in binary arithmetic 
	 * operations. It constrains templates to only work when the two 
	 * types may interoperate using the standard arithmetic binary 
	 * operators: +, -, *, /. For example, floats and ints are two 
	 * different types that may compose together in arithmetic binary 
	 * operations. The result may be of a type different than that of the 
	 * two parameters.
	 ********************************************************************/
	template<typename T, typename U>
	concept MutuallyArithmetic = requires (T t, U u)
	{
		requires Arithmetic<T>;
		requires Arithmetic<U>;

		{ t + u };
		{ t - u };
		{ t * u };
		{ t / u };
	};
}


//************************************************************************
#endif
