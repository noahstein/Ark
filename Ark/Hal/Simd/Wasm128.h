/*************************************************************************
 * @file
 * @brief SIMD tag and concept for the WebAssembly 128-bit SIMD ISA.
 *
 * @details The WebAssembly specfication adds an ISA implmenting 128-bit
 * SIMD operations, similar to the original SSE specification. This file
 * contains the definition of the tag and a concept for use by classes
 * and functions optimized with WebAssembly 128-bit SIMD data and
 * instructions.
 *
 * @sa @ref SimdArchitecture
 * @sa https://github.com/WebAssembly/simd
 * @sa https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md
 * @sa https://v8.dev/features/simd
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_WASM128_H_INCLUDE_GUARD)
#define ARK_HAL_SIMD_WASM128_H_INCLUDE_GUARD

#define SIMD_HAS_WASM128

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
	 * @brief SIMD architecture tag indicating the platform has the
	 * WebAssembly 128-bit SIMD ISA.
	 *
	 * @details Use this tag in template specialiazation parameter lists
	 * in class and struct definitions implemented in terms of the
	 * WebAssembly 128-bit SIMD specification.
	 *
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class Wasm128 : public None
	{
	};


	/*********************************************************************
	 * @brief Concept restricting templates to platforms with the
	 * WebAssembly 128-bit SIMD ISA.
	 *
	 * @details Use this concept when declaring template parameter lists
	 * for specializations of free functions optimized for the WebAssembly
	 * 128-bit SIMD ISA. Proeprly, its use will restrict overload
	 * resolution to only consider the function viable for platforms
	 * configured with the WebAssembly 128-bit SIMD ISA or future
	 * compatible revision.
	 *
	 * @see @ref SimdArchitecture
	 ********************************************************************/
	template<typename SIMD>
	concept Wasm128Family = std::is_base_of_v<Wasm128, SIMD>;
}


//========================================================================
#endif