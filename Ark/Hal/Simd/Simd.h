/*************************************************************************
 * @file
 * @brief SIMD specialization architexture
 * 
 * @details This file defines the small, top-level structure of the SIMD 
 * specialization architecture. It does two things:
 * 
 * 1. Checks to see the HAL_SIMD preprocessor macro has been defined and 
 * emits a preprocessor error if not. Without the macro, the specialized 
 * versions of the functions will not be found for the target CPU the 
 * source is being compiled for. Should a CPU not have a SIMD ISA, 
 * HAL_SIMD should be set equal to @ref ark::hal::simd::None.
 * 
 * 2. Defines the @ref INCLUDE_SIMD macro for platform-specific files 
 * to include optimized implementations specialized for specific SIMD 
 * hardware present in the configuration.
 * 
 * 3. Defines the class ark::hal::simd::None as a tag to be used when 
 * compiling for a CPU without SIMD instructions.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_HAL_SIMD_SIMD_H)
#define ARK_HAL_SIMD_SIMD_H


//************************************************************************
//  Dependencies
//************************************************************************
#include "../Hal.h"


//************************************************************************
//  Validation
//************************************************************************
#if !defined(HAL_SIMD)
#error You must define HAL_SIMD in the build configuration. Use 'None' if the target CPU has no SIMD support.
#endif


//************************************************************************
//  Platform-specific Inclusion Mechainism
//************************************************************************

/**
 * @brief Include a platform-specific SIMD header file
 * 
 * @details Any system that intends to include platform-specific 
 * implementations of data structures or algorithms based on the target 
 * platform's SIMD architecture shall include the SIMD-specific headers 
 * using this name resolution. Thus, the header files must conform to 
 * this naming convention. For example, an SSE2-optimized implementation 
 * of the Quat class would place that code in the file named`Quat_Sse2.h`.
 */
#define INCLUDE_SIMD(File) INCLUDE_HAL(HAL_SIMD, File)


//************************************************************************
//  Code
//************************************************************************
namespace ark::hal::simd
{
	/*********************************************************************
	 * @brief SIMD architecture tag indicating the CPU has no SIMD ISA.
	 *
	 * @details Algorithm implementations should not use this tag. The 
	 * various ags are used to mark specializations of templatized 
	 * algorithms. Unspecialized versions shall be written as the general 
	 * template case to be specialized for ISAs with the use of the other 
	 * tags.
	 * 
	 * The one valid use of this tag outside of specialized 
	 * implementations is to ensure testing of the base, geenric 
	 * algorithms. This ensures that no attemtp is made to use any of 
	 * the specialization machinary.
	 *
	 * Tags for the first revision of SIMD architectures shall derive 
	 * from this tag to create a unified hierarchy of SIMD ISA tags. As 
	 * there is no commonality between different SIMD ISAs such as SSE, 
	 * Neon, and AltiVec, this is the common base.
	 * 
	 * @sa @ref SimdArchitecture
	 ********************************************************************/
	class None
	{
	};
}



//************************************************************************
//  Platform-optimized Specializations
//************************************************************************
#if __has_include(INCLUDE_HAL_LOCAL(HAL_SIMD))
#include INCLUDE_HAL_LOCAL(HAL_SIMD)
#endif


//************************************************************************
#endif
