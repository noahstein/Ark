/*************************************************************************
 * @file
 * @brief Definitions for the Vec primary vector class
 * 
 * @details Define the a class and supporting constructs to define a 
 * standard dense vector. The class defined here is not designed for 
 * general-purpose linear algebra applications. Its design  is oriented 
 * for use in graphical applications, be they 2D or 3D. The design is 
 * composed of a few components:
 * 
 * - Vec: A template type to use when needing a vector for a graphics 
 *   application is needed. The vector has template parameters for 
 *   specifying the type and number of elements.
 * 
 * - VecBasic: A simple default implementation of a Vec. This class is 
 *   portable to all platforms as long as the chosen scalar type is 
 *   available. It relies upon the general vector expressions for its 
 *   functionality.
 * 
 * - VectorSelector: A support class used by Vec to select which 
 *   vector class to use for a Vec of given type(s). It is necessary due 
 *   to no partial specialization of template using definitions.
 * 
 * - VecNone: A concept used in overload resolution. It is used as a 
 *   base class for SIMD ISA concepts. These define SIMD requirements for 
 *   function signatures that then restrict the applicability of 
 *   functions based on the SIMD revision.
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VEC_H_INCLUDE_GUARD)
#define ARK_MATH_VEC_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include "Vector.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*********************************************************************
	 * @brief Vector Concept with No SIMD Optimization
	 *
	 * @tparam V The vector class under consideration
	 * @tparam S The required type of the scalars in V
	 * @tparam N The required dimension of the vector V
	 *
	 * @details The concept of a vector with no SIMD optimization is
	 * essentially the concept of a vector of a given scalar type and
	 * dimension. All SIMD-optimized vector concepts should refer to 
	 * this one either directly or indirectly.
	 *
	 * @sa ::ark::math::Vector
	 * @sa @ref SimdArchitecture
	 */
	template<typename V, typename S, std::size_t N>
	concept VecNone = 
		Vector<V> &&
		std::same_as<typename V::Scalar, S> &&
		OfKnownDimension<V, N>;


	/*********************************************************************
	 * @brief Basic N-dimensional Vector
	 * 
	 * @tparam S The type of the scalar values
	 * @tparam N The dimension (number of elements)
	 * 
	 * @details This class defines a simple and general template 
	 * implementation of the Vector concept. Its footprint is a dense 
	 * C-array of data. For architectures with SIMD ISAs, it is fully 
	 * expected other classes to define appropriate storage classes and 
	 * optimized functions.
	 *
	 * @sa ::ark::math::Vector
	 * @sa ::ark::math::VecNone
	 * @sa SimdConceptHierarchy
	 */
	template<typename S, std::size_t N>
	class VecBasic
	{
	public:
		/// @brief The Scalar type required by the @ref ::ark::math::Vector "Vector" concept
		using Scalar = S;

	private:
		/// @brief Data storage is a simple C-style array
		Scalar data_[N];

	public:
		/*********************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/GenericDefaultConstructor.txt
		 */
		constexpr VecBasic() noexcept(std::is_nothrow_constructible_v<Scalar>) = default;


		/*********************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		template<typename ... T>
			requires (sizeof...(T) == N)
		constexpr VecBasic(T && ... values) noexcept((std::is_nothrow_convertible_v<T, Scalar> && ...))
		{
			auto Set = [this, i = 0] <typename U> (U && value) mutable
			{
				data_[i++] = static_cast<Scalar>(std::forward<U>(value));
			};

			(Set(values), ...);
		}


		/*********************************************************************
		 * @brief Vector Constructor
		 * @details @includedoc Math/Vector/VectorConstructor.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && SameDimension<VecBasic, V>
		constexpr VecBasic(const V& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			std::size_t length = Size();
			for (std::size_t i = 0; i < length; ++i)
			{
				data_[i] = static_cast<Scalar>(rhs(i));
			}
		}


		/*********************************************************************
		 * @brief Vector Assignment
		 * @details @includedoc Math/Vector/VectorAssignment.txt
		 */
		template<Vector V>
			requires std::convertible_to<typename V::Scalar, Scalar> && SameDimension<VecBasic, V>
		constexpr VecBasic& operator=(const VecBasic& rhs) noexcept(std::is_nothrow_convertible_v<typename V::Scalar, Scalar>)
		{
			std::size_t length = Size();
			for (std::size_t i = 0; i < length; ++i)
			{
				data_[i] = static_cast<Scalar>(rhs(i));
			}
			return *this;
		}


		/*********************************************************************
		 * @brief The number of elements in the vector, its dimension
		 */
		static constexpr size_t Size() noexcept { return N; }


		/*********************************************************************
		 * @brief Element Accessor
		 */
		constexpr Scalar operator()(size_t index) const noexcept(std::is_nothrow_copy_constructible_v<Scalar>)
		{
			return data_[index];
		}
	};


	/*********************************************************************
	 * @brief Vec Selector Template
	 *
	 * @tparam S The type of the scalar components
	 * @tparam N The dimension of the vector
	 * @tparam I The SIMD architecture
	 *
	 * @details The template defines a single type (named "type"!) to
	 * define which vector class should be used given the scalar type
	 * S, the dimension, and SIMD architecture I. WHen a SIMD ISA provides 
	 * facilities to optimize algorithms, create any new class and function 
	 * implementations for that ISA.Then define a specialization of this 
	 * template to enable Vec to select it when that implementation is 
	 * appropriate.
	 ********************************************************************/
	template<typename S, std::size_t N, typename I>
	struct VectorSelector
	{
		using type = VecBasic<S, N>;
	};


	/*********************************************************************
	 * @brief Standard Dense Vector
	 *
	 * @tparam S The type of the scalar components
	 * @tparam N The dimension of the vector
	 * @tparam I The SIMD architecture
	 *
	 * @details Programmers who desire to use a "normal" vector should
	 * use this type alias. It is designed to automatically pick the best
	 * class based on the scalar type and the SIMD ISA, falling back to
	 * the default implementation VecBasic when there is no better SIMD
	 * ISA. To do this, it uses VectorSelector to select the optimal type 
	 * based on the template parameters. THe I template parameter is 
	 * generally left unspecified in order to select the SIMD ISA 
	 * specified in the build configuration.
	 ********************************************************************/
	template<typename S, std::size_t N, typename I = ark::hal::simd::HAL_SIMD>
	using Vec = typename VectorSelector<S, N, I>::type;
}

/*========================================================================
	Platform-optimized Specializations
========================================================================*/
#if __has_include(INCLUDE_SIMD(Vec))
#include INCLUDE_SIMD(Vec)
#endif


//========================================================================
#endif
