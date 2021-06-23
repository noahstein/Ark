/*========================================================================
Description
	Definee fundamental mathematical concepts used by library classes, 
	but not "part" of those classes, i.e. the Arithmetic concept is 
	defined here; however, the Vector concept is defined in the header 
	with the Vector-rleated class definitions.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_CONCEPTS_H_INCLUDE_GUARD)
#define ARK_MATH_CONCEPTS_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include <type_traits>


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
		Arithmetic concept defines a type that may undergo standard 
		arithmetic unary and binary operations with one or two objects of 
		that type.
	--------------------------------------------------------------------*/
	template<typename T>
	concept Arithmetic = std::is_arithmetic_v<T>;


	/*--------------------------------------------------------------------
		The MutuallyArithmetic concept defines when two individually 
		arithmetic types may validly be used together in arithmetic 
		expressions. This checks for situations such as floats & ints 
		being used together, etc.
	--------------------------------------------------------------------*/
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


//========================================================================
#endif
