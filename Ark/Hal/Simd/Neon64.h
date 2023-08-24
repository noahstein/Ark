/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the ARM Neon AArch64 ISA.
 *
 * @details The AArch64 implementation of the Neon ISA adds new 
 * instructions to streamline some 32-bit 2- and 4-element 
 * single-precision floating-point algorithms. It also adds 2-element 
 * 64-bit double-precision floating-point functionality.
 *
 * @sa @ref SimdArchitecture
 * @sa https://www.arm.com/technologies/neon
 * @sa https://community.arm.com/arm-community-blogs/b/operating-systems-blog/posts/arm-neon-programming-quick-reference
 * @sa https://developer.arm.com/architectures/instruction-sets/intrinsics/
 * @sa https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_NEON64_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_NEON64_H_INCLUDE_GUARD

#define SIMD_HAS_NEON64


//************************************************************************
// Dependencies
//************************************************************************
#include "Neon32.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the ARM Neon 
	 * AArch64 ISA.
	 *
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for Neon64-specific optimizations.
	 *
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Neon64 : public Neon32
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the ARM Neon
	 * AArch64 ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for Neon64-optimized specializations of free functions. It will 
	 * properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the Neon64 or other 
	 * compatible SIMD ISA.
	 *
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Neon64Family = Neon32Family<SIMD> && std::derived_from<SIMD, Neon64>;
}


//************************************************************************
#endif
