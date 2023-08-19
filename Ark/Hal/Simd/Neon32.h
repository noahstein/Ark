/*========================================================================
Description
	SIMD tag for the ARM Neon 32-bit ISA.

Copyright
	Copyright (c) 2021-2023 Noah Stein. All Rights Reserved.
========================================================================*/

/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the ARM Neon AArch32 ISA.
 *
 * @details The ARM RISC architecture added floating-point SIMD registers 
 * and instructions with its technology named Neon. It added features 
 * similar to SSE in the Intel x86 world. In a subsequent update, the 
 * original Neon was given the appelation AArch32 to differentiate it 
 * from the enhanced feature set.
 *
 * @sa @ref SimdArchitecture
 * @sa https://www.arm.com/technologies/neon
 * @sa https://community.arm.com/arm-community-blogs/b/operating-systems-blog/posts/arm-neon-programming-quick-reference
 * @sa https://developer.arm.com/architectures/instruction-sets/intrinsics/
 * @sa https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
 *
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_NEON32_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_NEON32_H_INCLUDE_GUARD


//========================================================================
//	Dependencies
//========================================================================
#include "Simd.h"


//========================================================================
//	Code
//========================================================================
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has the ARM Neon
	 * AArch32 ISA.
	 *
	 * @details Use this tag in template specialiazation parameter lists 
	 * in class and struct definitions for 32-bit Neon-specific 
	 * optimizations.
	 *
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Neon32 : public None
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to CPUs with the ARM Neon
	 * AArch32 SIMD ISA.
	 *
	 * @details Use this concept when declaring template parameter lists 
	 * for 32-bit Neon-optimized specializations of free functions. It 
	 * will  properly restrict overload resolution to only consider the 
	 * function viable for platforms configured with the 32-bit Neon or 
	 * other compatible SIMD ISA.
	 *
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Neon32Family = std::is_base_of_v<Neon32, SIMD>;
}


//========================================================================
#endif