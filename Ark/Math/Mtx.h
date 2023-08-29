/*************************************************************************
 * @file
 * @brief Definitions for the Mtx primary matrix class
 *
 * @details Define the a class and supporting constructs to define a
 * standard dense matrix. The class defined here is not designed for
 * general-purpose linear algebra applications. Its design  is oriented
 * for use in graphical applications, be they 2D or 3D. The design is
 * composed of a few components:
 *
 * - Vec: A template type to use when needing a vector for a graphics
 *   application is needed. The vector has template parameters for
 *   specifying the type and number of elements.
 *
 * - MtxBasic: A simple default implementation of a Mtx. This class is
 *   portable to all platforms as long as the chosen scalar type is
 *   available. It relies upon the general vector expressions for its
 *   functionality.
 *
 * - MatrixSelector: A support class used by Vec to select which
 *   matrix class to use for a Mtx of given type(s). It is necessary due
 *   to no partial specialization of template using definitions.
 *
 * - MtxNone: A concept used in overload resolution. It is used as a
 *   base class for SIMD ISA concepts. These define SIMD requirements for
 *   function signatures that then restrict the applicability of matrix 
 *   functions based on the SIMD revision.
 *
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_MTX_H_INCLUDE_GUARD)
#define ARK_MATH_MTX_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include "Matrix.h"
#include "Ark/Hal/Simd/Simd.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*********************************************************************
	 * @brief Matrix Concept with No SIMD Optimization
	 *
	 * @tparam M The matrix class under consideration
	 * @tparam S The required type of the scalars in M
	 * @tparam R The required number of rows in the matrix M
	 * @paramt C The required number of columns in the matrix M
	 *
	 * @details The concept of a matrix with no SIMD optimization is
	 * essentially the concept of a matrix of a given scalar type and
	 * dimension. All SIMD-optimized matrx concepts should refer to
	 * this one either directly or indirectly.
	 *
	 * @sa ::ark::math::Matrix
	 * @sa @ref SimdArchitecture
	 */
	template<typename M, typename S, std::size_t R, std::size_t C>
	concept MtxNone =
		Matrix<M> &&
		std::same_as<typename M::Scalar, S> &&
		MatrixOfKnownSize<M, R, C>;


	/*********************************************************************
	 * @brief Basic Matrix of RxC dimensions
	 *
	 * @tparam S The type of the scalar values
	 * @tparam R The number of rows in the matrix
	 * @tparam C The number of columns in the matrix
	 *
	 * @details This class defines a simple and general template
	 * implementation of the Matrix concept. Its footprint is a dense
	 * two-dimension C-array of data. For architectures with SIMD ISAs, 
	 * it is fullyexpected other classes to define appropriate storage 
	 * classes and optimized functions.
	 *
	 * @sa ::ark::math::Matrix
	 * @sa ::ark::math::MtxNone
	 * @sa @ref SimdConceptHierarchy
	 */
	template<typename S, std::size_t  R, std::size_t C>
	class MtxBasic
	{
	public:
		using Scalar = S;

	private:
		/// @brief Data stored in a simple two-dimensional C-style array
		S data_[R][C];

	public:
		/// @name Constructors
		/// @{

		/*********************************************************************
		 * @brief Default Constructor
		 * @details @includedoc Math/Vector/GenericDefaultConstructor.txt
		 */
		MtxBasic() = default;


		/*********************************************************************
		 * @brief Scalar Constructor
		 * @details @includedoc Math/Vector/ScalarConstructor.txt
		 */
		template<typename ... T>
			requires (sizeof...(T) == R * C)
		constexpr MtxBasic(T && ... values)
		{
			auto Set = [this, r = 0, c = 0] <typename T> (T && value) mutable
			{
				data_[r][c] = static_cast<Scalar>(std::forward<T>(value));
				if (++c == Width())
				{
					c = 0;
					++r;
				}
			};

			(Set(values), ...);
		}

		/// @}

		/*********************************************************************
		 * @brief Matrix Assignment
		 * @details @includedoc Math/Matrix/MatrixAssignment.txt
		 */
		template<Matrix M>
			requires std::convertible_to<typename M::Scalar, Scalar>
				&& SameSize<M, MtxBasic>
		constexpr MtxBasic& operator=(const M& rhs)
		{
			for (std::size_t r = 0; r < Height(); ++r)
			{
				for (std::size_t c = 0; c < Width(); ++c)
				{
					data_[r][c] = rhs(r, c);					
				}
			}
			return *this;
		}


		/// @name Accessors 
		/// @{

		/*********************************************************************
		 * @brief Get the number of columns in the matrix
		 */
		static constexpr std::size_t Width() { return C; }


		/*********************************************************************
		 * @brief Get the number of rows in the matrix
		 */
		static constexpr std::size_t Height() { return R; }

		/*********************************************************************
		 * @brief Element Accessor
		 */
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		 {
			  return data_[row][column];
		}

		/// @}
	};


	/*********************************************************************
	 * @brief Mtx Selector Template
	 *
	 * @tparam S The type of the scalar components
	 * @tparam R The number of rows in the matrix
	 * @tparam C The number of columns in the matrix
	 * @tparam I The SIMD architecture
	 *
	 * @details The template defines a single type (named "type"!) to
	 * define which matrix class should be used given the scalar type
	 * S, the height, the width, and SIMD architecture I. WHen a SIMD ISA 
	 * provides facilities to optimize algorithms, create any new class 
	 * and function implementations for that ISA.Then define a 
	 * specialization of this template to enable Vec to select it when 
	 * that implementation is appropriate.
	 ********************************************************************/
	template<typename S, std::size_t R, std::size_t C, typename IC>
	struct MatrixSelector
	{
		using type = MtxBasic<S, R, C>;
	};


	/*********************************************************************
	 * @brief Standard Dense Matrix
	 *
	 * @tparam S The type of the scalar components
	 * @tparam R The number of rows in the matrix
	 * @tparam C The number of columns in the matrix
	 * @tparam I The SIMD architecture
	 *
	 * @details Programmers who desire to use a "normal" matrix should
	 * use this type alias. It is designed to automatically pick the best
	 * class based on the scalar type and the SIMD ISA, falling back to
	 * the default implementation MtxBasic when there is no better SIMD
	 * ISA. To do this, it uses VectorSelector to select the optimal type
	 * based on the template parameters. The I template parameter is
	 * generally left unspecified in order to select the SIMD ISA
	 * specified in the build configuration.
	 ********************************************************************/
	template<typename S, std::size_t R, std::size_t C, typename I = ark::hal::simd::HAL_SIMD>
	using Mtx = typename MatrixSelector<S, R, C, I>::type;
}



/*========================================================================
	Platform-optimized Specializations
========================================================================*/
#if __has_include(INCLUDE_SIMD(Mtx))
#include INCLUDE_SIMD(Mtx)
#endif


//========================================================================
#endif
