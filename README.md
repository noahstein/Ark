# Ark
Musings on a modern C++ game engine architecture. It's currently a long, long way from doing anything. It's merely a test-bed for seeing what 
I think would be a good way to implement an engine from scratch in a post-C++20 world.

## Building
The project had been using Bazel; however, it is now using CMake.

## Engine Design
There are a few major aims of the project:

* **Mult-platform Code**. The majority of the code should be identical on all platforms.
* **Optimized Code**. The code should be highly-optimized for the platforms it runs on. There's no point in attempting to write multi-platform code if it runs efficiently on only one—or possibly no—system. The architecture is designed to isolate optimizations from polluting platform-independent code.
* **Clean code**. Codebases too often are loaded up with all kinds of conditional compilation blocks for various reasons: select certain header files in certain circumstances, call platform-specific functions, or little blocks of build-specific code. A major objective is to minimize the use of the proprocessor, especially the use of `#ifdef`. The goal of no `#ifdef` blocks might not be reached, but they will certainly be minimized more so than with other engines.

### Feature-specific Include Files

The use of `ifdef` is avoided even in bringing in platform- and feature-specific include files. For example, this is the last line of the file *Quat.h*:
```
#include INCLUDE_SIMD(Quat)
```
It includes the platform-specific implementation of a quaternion based on its SIMD architecture. The `INCLUDE_SIMD` macro builds out the appropriate header file name based on the build configuration. For example, if the build SIMD ISA is SSE2, *Quat.h* will include the file *Quat_Sse2.h*, which includes optimized quaternion math functions implemented using SSE2 compiler intrinsics.


### Platform-specific Optimizated Classes and Functions

To see how the code handles per-platform optimization, examine the quaternion math files: *Quaternion.h*, *Q_uat.h*, *Quat_[].h*.

*Quaternion.h* defines the concept of a quaternion and platform-independent reference implementations of the algebra. Concepts a are a new feature in C++ 20. Please see an online reference [Constraints and Concepts](https://en.cppreference.com/w/cpp/language/constraints) at [C++ Reference](https://en.cppreference.com/w/). This is an abstract notion of what a quaternion is. In C++ now, it permits duck typing in template code. It's useful in multi-platform development as it allows specifying how a type must act to be usable in client code. Then platform-specific classes and functions may be written to the concept, and you may be sure it works (as long as it actually implements the desired behavior on the platform.) *QuaternionUnitTests.cpp* contains tests of the abstract algebra. 

The C++ concept `Quaternion` is very small. Any class that has a public typedef of its `Scalar` type (the type of the elements) and the public const functions `w()`, `x()`, `y()`, and `z()` returning values of `Scalar` type is a valid quaternion. Mathematically, a quaternion has a defined algebra: addition, multiplication, additive & mulitiplicative inverses, conjugate, etc. In this code, the algebraic functions are implemented using the quaternion concept, not any specific concrete type. Thus, any new type that meets the tiny concept of quaternion will automatically participate in the full algebra.  Additionally, it will do so with any other quatnerion type that has a compatible scalar type. It just works™.

With the concept in place, here is the declaration of the canonical quaternion class in the library:

```
template<typename S, typename I = ark::hal::HAL_SIMD> class Quat
```

First, notice it is a template class. There are two template parameters:

1. **S**: The type of the scalars. This will get used as the `Scalar` typedef.
2. **I**: The SIMD instruction set architecture. It defaults to `ark::hal::HAL_SIMD`. `HAL_SIMD` is not a specific architecture. It is a preprocessor macro the build system sets to the name of the specific architecture.

So, the template quaternion class may be specialized by both the type of the numbers (int, short, float, double, etc.) as well as the instruction architecture. The macro `HAL_SIMD` isn't just a preprocessor macro like in other libraries. It is used by the `INCLUDE_SIMD` macro to pick out which header to include; however, it is more than that. The HAL (hardware abstraction layer) defines classes with the names of the SIMD architectures: Sse, Sse2, Sse3, etc. define the various generations of SSE ISAs. These classes are defined through derivation, so Sse is the base class of Sse2, which is the base class of Sse3, and so on. What does this give us? Plenty.

When the architecture is Sse, *Quat_Sse.h* is included whenever *Quat.h* is included. *Quat_Sse.h* defines a specialization Quat declared like this:

```
template<> class Quat<float, ark::hal::Sse>

```

The body of the class contains an SSE-specific data type and implements the quaternion algebra using SSE compiler intrinsics. When client code defines a `Quat<float>`, the SSE architecture will automatically get picked up from the default through the macro, and the quaternion will use optimized unary operations for negation, conjugate, etc. If the other qaternion in a binary operation is also a `Quat<float>`, the operation will also use optimized SSE instructions. Whenever a `Quat<float>` is used in a binary operation with a type that meets the `Quaternion` concept but is not a `Quat<float>`, it will still be compiled and computed, it just won't be SSE optimized.

Say SSE2 added a few instructions that makes one or two operations more efficient, but doesn't affect 95% of the code base. Here's where the class-based solution really shines. SSE2 is implemented as derived from SSE, so only the SSE2-specific functions need to be written, and the SSE one operations will get inherited almost for free. Please see *Quat_Sse2.h* for a contrived example to show just how easy it is to extend to a new generation of architectures.


## Paradigm Branches

The are currently two platform-independence paradigms designed with at least partial implementations of the basic math types vector, matrix, and quaternion. Additionally, there is an idea for a third paradigm that is a hybrid of the first two; hwoever, it has not yet been implemented yet. The three paradigms are:

1. Template Selection: Select the SIMD-optimized via a using type alias. On branch math_template_selection
2. Template Specialization: The SIMD-optimized impelementation is defined in a specialization of a single template class. On branch math_template_specialization
3. Template Specialization via Inheritence: The SIMD-optimized implementation is written in its own class that is then used as a base class in the definition of template class specializsations. To be on branch math_template_specialization_via_inheritence
