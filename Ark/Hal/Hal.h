/*************************************************************************
 * @file
 * @brief Hardware Abstraction Layer
 * 
 * @details This file contains some basic definitions for the Hardware 
 * Abstraction Lyaer (HAL). The HAL umbrealla encompasses anything that 
 * abstracts any and all platform-specific aspects for the hardware.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/
 
#if !defined(ARK_HAL_H_INCLUDE_GUARD)
#define ARK_HAL_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include "Ark/Hal/Hal.h"


//************************************************************************
//  Platform-specific Inclusion Mechainism
//************************************************************************

/**
 * @internal
 * @brief Add quatoation marks around a string in the preprocessor.
 * 
 * @details Place quotation marks around the text of the argument to 
 * create a C-style string of it. Due to the C preprocessor substitution 
 * rules, this must be a separate macro invocation to ensure the expected 
 * and necessary token substitutions to occur. 
 */
#define INCLUDE_STRINGIZE(String) #String


/**
 * @internal
 * @brief Build up a platform-specific include file name.
 * 
 * @param Part The type of platform-specific file to include, e.g. Cpu, 
 * Simd, Os, etc.
 * 
 * @param File The base file name of the platform-specifc version to be 
 * included.
 * 
 * @details Geenerates the a platform-specific header file name, given 
 * the platform-independence zone amd base file name. Thus, for example, 
 * when including the SSE2 optimizations for the Quat implementation, at 
 * this stage of macro expansion, the invocation of:
 * ```
 * INCLUDE_BUILD_FILENAME(Sse2, Quat)
 * ```
 * will resolve into the filename:
 * ```
 * Quat_Sse2.h
 * ```
 * WHen used in place of a simple file name in an include directive, the 
 * preprocessor will include the correct platform-specific header file.
 * 
 * @note This should almost never be used directly by header files. It is 
 * intended that each module (Cpu, Simd, etc.) will define in its 
 * interface a module-specific directive, one that will then rely upon 
 * this macro in its definition.
 */
#define INCLUDE_BUILD_FILENAME(Part, File) INCLUDE_STRINGIZE(File ## _ ## Part.h)


/**
 * @internal
 * @brief Include a local file based on platform config
 * 
 * @param Part The type of platform-specific definition to include, 
 * such as CPU, SIMD, etc.
 * 
 * @details This macro is used to load a file with the name of the 
 * configuration value only. This is useful for defining constants, 
 * classes, functions, and whatever else necessary to define that 
 * specific version of hardware. For example,
 * ```
 * INCLUDE_HAL_LOCAL(HAL_SIMD)
 * ```
 * could result in the text:
 * ```
 * Sse2.h
 * ```
 * And this  is useful in the Simd directory to load the definitions of 
 * each type of SIMD hardware.
 */
#define INCLUDE_HAL_LOCAL(Part) INCLUDE_STRINGIZE(Part.h)


/**
 * @internal
 * @brief Construct the name of a platform-specific hardware abstraction 
 * layer header file.
 * 
 * @param Part The type of the platform-specific file, such as CPU, SIMD, 
 * etc.
 * 
 * @param File The base file name of the platform-specific header 
 * whose name is to be generated.
 * 
 * @details This macro is used in include directives in platform-
 * independent header files (or even source files) to include 
 * platform-specific header files for the current configuration 
 * getting compiled. For example:
 * ```
 * #include INCLUDE_HAL(HAL_SIMD, Quat)
 * ```
 * could result after macro expansion to the text:
 * ```
 * #include "Quat_Sse2.h"
 * ```
 * resulting in the platform-independent file `Quat.h` including the data 
 * classes and functions optimized for the SSE2 architecture.
 * 
 * @note This macro is not expected to be used directly by applicsation 
 * code beyond the implementation of macros to directly include platform-
 * specific files but instead to be used as the basis of inclusion macros 
 * for each type of abstracted hardware. For example, the SIMD zone 
 * defines its own inclusion mechanism `@ref INCLUDE_SIMD`. That macro 
 * uses `INCLUDE_HAL` in its body, but the SIMD application code never does 
 * directly.
 */
#define INCLUDE_HAL(Part, File) INCLUDE_BUILD_FILENAME(Part, File)


//************************************************************************
#endif
