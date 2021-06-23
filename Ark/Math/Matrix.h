/*========================================================================
Description
	Defines the Matrix concept to represent the abstract definition
	of the structure of a mathematical matrix. From there, this file 
	implements many common vector functions on that abstraction. Due 
	to this implementation to the abstraction, each new concrete 
	matrix class oinly needs to define the type of its scalars and 
	accessors to its elements, and it will not only automatically have 
	a full set of operations, it will interact with any other mix of 
	matrix classes, as long as the scalar types may interact 
	arithmetically.
	
	Additionally, any specific matrix class may define its own 
	operators and implement optimized code for those oeprations. Classes 
	may implement any operators they so choose and still rely upon the 
	abstract implementations for the rest.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_MATRIX_H_INCLUDE_GUARD)
#define ARK_MATH_MATRIX_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "Concepts.h"


/*========================================================================
	Code
========================================================================*/
namespace ark::math
{
	/*====================================================================
		Concepts
	====================================================================*/

	/*--------------------------------------------------------------------
		@brief Matrix Concept

		@details The Vector concept defines an abstract view of a vector.
		Please note this concept is a subset of the mathematical 
		definition of a vector. The mathematical definition includes the 
		basic arithmetic operations: addition, subtraction,  
		multiplication by a scalar, etc. Those are no defined here. This 
		concept models the definition of a vector as an array of numbers.
		Even the definitional arithmetic operations are implemented 
		against this definition thus providing maximum reuse and 
		minimizing the amount of unnecessary duplicated code.

		As such, a class models a vector if it meets 4 criteria:

		1) It defines a type Scalar to indicate the type of all its 
		components. The Scalar must be an arithmetic type as the 
		mathematical concept defines addition and multiplication of
		it components.

		2) It defines the static member function Width() and Height() to 
		define the size of the matrix.

		3) It has an operator() to return the values of its components.
	--------------------------------------------------------------------*/
	template<typename M>
	concept Matrix = requires(M m)
	{
		typename M::Scalar;
		Arithmetic<typename M::Scalar>;

		{ M::Width() } -> std::same_as<std::size_t>;
		{ M::Height() } -> std::same_as<std::size_t>;
		{ m(1, 1) } -> std::same_as<typename M::Scalar>;
	};


	/*--------------------------------------------------------------------
		@brief SameSize Concept

		@details Some binary matrix operations require the two matrices be 
		of the same size.
	--------------------------------------------------------------------*/
	template<typename ML, typename MR>
	concept SameSize  = requires
	{
		requires (ML::Width() == MR::Width());
		requires (ML::Height() == MR::Height());
	};


	/*--------------------------------------------------------------------
		@brief MatrixMultiplicationSizes Concept

		@details Matrix multiplication is a binary operation where the 
		two matrices may have different sizes; however, the sizes must 
		conform in a certain way. They must take the form of an I x J 
		matrix times a J x K matrix. That is, the number of columns in 
		the first matrix must be the same as the number of rows in the 
		second.
	--------------------------------------------------------------------*/
	template<typename ML, typename MR>
	concept MatrixMultiplicationSizes = requires
	{
		requires (ML::Width() == MR::Height());
	};


	/*--------------------------------------------------------------------
		@brief SquareSize Concept

		@details The width and height of a matrix class are the same, 
		resulting in a square shape. Certain algorithms are only valid on 
		square matrices.
	--------------------------------------------------------------------*/
	template<typename M>
	concept SquareSize = requires
	{
		requires (M::Width() == M::Height());
	};


	/*--------------------------------------------------------------------
		@brief MatrixOfKnownSize Concept

		@details Restrict to a matrix of a given size. This is useful for 
		overloads with different implementations based on size, e.g. the 
		multiple functions for computing the determinant.
	--------------------------------------------------------------------*/
	template<typename M, int H, int W>
	concept MatrixOfKnownSize = requires
	{
		requires (M::Height() == H);
		requires (M::Width() == W);
	};


	/*====================================================================
		Expression Base Classes
	====================================================================*/

	/*--------------------------------------------------------------------
		Base class of all matrix expressions
	--------------------------------------------------------------------*/
	class MatrixExpr
	{
	};


	/*--------------------------------------------------------------------
		Base class of 1-argument matrix expressions
	--------------------------------------------------------------------*/
	template<Matrix M>
	class MatrixUnaryExpr : public MatrixExpr
	{
	public:
		using Scalar = typename M::Scalar;
	};


	/*--------------------------------------------------------------------
		Base class of 2-argument vector expressions
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
	requires MutuallyArithmetic<typename ML::Scalar, typename MR::Scalar>
	class MatrixBinaryExpr : public MatrixExpr
	{
		using SL = typename ML::Scalar;
		using SR = typename MR::Scalar;

	public:
		using Scalar = typename std::common_type<SL, SR>::type;
	};


	/*====================================================================
		Expression Function Classes
	====================================================================*/
	
	/*--------------------------------------------------------------------
		Matrix Negation Expression: -m
	--------------------------------------------------------------------*/
	template<Matrix M>
	class MatrixNegation : public MatrixUnaryExpr<M>
	{
		M const & m_;

	public:
		using Scalar = typename MatrixUnaryExpr<M>::Scalar;

		constexpr MatrixNegation(const M& m) noexcept
			: m_(m)
		{}

		static constexpr size_t Width() noexcept { return M::Width(); }
		static constexpr size_t Height() noexcept { return M::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			return -m_(row, column);
		}
	};


	/*----------------------------------------------------------------
		Matrix Negation Operator: -m
	----------------------------------------------------------------*/
	template<Matrix M>
	inline constexpr auto operator-(const M& m) noexcept -> MatrixNegation<M>
	{
		return MatrixNegation(m);
	}


	/*--------------------------------------------------------------------
		Matrix Addition Expression: m1 + m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires SameSize<ML, MR>
	class MatrixAddition : public MatrixBinaryExpr<ML, MR>
	{
		using Expr = typename MatrixBinaryExpr<ML, MR>;

		ML const & l_;
		MR const & r_;

	public:
		using Scalar = Expr::Scalar;

		constexpr MatrixAddition(const ML& lhs, const MR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		static constexpr size_t Width() noexcept { return ML::Width(); }
		static constexpr size_t Height() noexcept { return ML::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			return l_(row, column) + r_(row, column);
		}
	};


	/*--------------------------------------------------------------------
		Matrix Addition Operator: m1 + m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires SameSize<ML, MR>
	inline constexpr auto operator+(const ML& lhs, const MR& rhs) noexcept -> MatrixAddition<ML, MR>
	{
		return MatrixAddition(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		Matrix Subtraction Expression: m1 - m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires SameSize<ML, MR>
	class MatrixSubtraction : public MatrixBinaryExpr<ML, MR>
	{
		using Expr = typename MatrixBinaryExpr<ML, MR>;

		ML const & l_;
		MR const & r_;

	public:
		using Scalar = Expr::Scalar;

		constexpr MatrixSubtraction(const ML& lhs, const MR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		static constexpr size_t Width() noexcept { return ML::Width(); }
		static constexpr size_t Height() noexcept { return ML::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			return l_(row, column) - r_(row, column);
		}
	};


	/*--------------------------------------------------------------------
		Matrix Subtraction Operator: m1 - m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires SameSize<ML, MR>
	inline constexpr auto operator-(const ML& lhs, const MR& rhs) noexcept -> MatrixSubtraction<ML, MR>
	{
		return MatrixSubtraction(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		Matrix Scalar Multiplication Expression: s * m, m * s
	--------------------------------------------------------------------*/
	template<Matrix M, typename S>
	class MatrixScalarMultiplication : public MatrixExpr
	{
		M const & m_;
		S s_;

	public:
		using Scalar = std::common_type<typename M::Scalar, S>::type;

		constexpr MatrixScalarMultiplication(const M& m, S s) noexcept
			: m_(m), s_(s)
		{}

		static constexpr size_t Width() noexcept { return M::Width(); }
		static constexpr size_t Height() noexcept { return M::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			return s_ * m_(row, column);
		}
	};


	/*--------------------------------------------------------------------
		Scalar-Matrix Multiplication Operator: s * m
	--------------------------------------------------------------------*/
	template<typename S, Matrix M>
		requires MutuallyArithmetic<S, typename M::Scalar>
	inline constexpr auto operator*(S lhs, const M& rhs) noexcept -> MatrixScalarMultiplication<M, S>
	{
		return MatrixScalarMultiplication(rhs, lhs);
	}


	/*--------------------------------------------------------------------
		Matrix-Scalar Multiplication Operator: m * s
	--------------------------------------------------------------------*/
	template<Matrix M, typename S>
		requires MutuallyArithmetic<S, typename M::Scalar>
	inline constexpr auto operator*(const M& lhs, S rhs) noexcept -> MatrixScalarMultiplication<M, S>
	{
		return MatrixScalarMultiplication(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		Matrix Scalar Division Expression: m / s
	--------------------------------------------------------------------*/
	template<Matrix M, typename S>
	class MatrixScalarDivision : public MatrixExpr
	{
		M const & m_;
		S s_;

	public:
		using Scalar = std::common_type<typename M::Scalar, S>::type;

		constexpr MatrixScalarDivision(const M& m, S s) noexcept
			: m_(m), s_(s)
		{}

		static constexpr size_t Width() noexcept { return M::Width(); }
		static constexpr size_t Height() noexcept { return M::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			return m_(row, column) / s_;
		}
	};


	/*--------------------------------------------------------------------
		Matrix-Scalar Division Operator: m / s
	--------------------------------------------------------------------*/
	template<Matrix M, typename S>
	inline constexpr auto operator/(const M& lhs, S rhs) noexcept -> MatrixScalarDivision<M, S>
	{
		return MatrixScalarDivision(lhs, rhs);
	}

	/*--------------------------------------------------------------------
		Matrix Equalaity Operator: m1 == m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires SameSize<ML, MR>
	inline constexpr auto operator == (const ML& lhs, const MR& rhs) noexcept -> bool
	{
		for (std::size_t r = 0; r < ML::Height(); ++r)
		{
			for (std::size_t c = 0; c < ML::Width(); ++c)
			{
				if (lhs(r, c) != rhs(r, c))
				{
					return false;
				}
			}
		}
		return true;
	}


	/*--------------------------------------------------------------------
		Matrix Multiplication Expression: m1 * m2
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
		requires MatrixMultiplicationSizes<ML, MR>
	class MatrixMultiplication : public MatrixBinaryExpr<ML, MR>
	{
		ML const & l_;
		MR const & r_;

	public:
		using Scalar = typename MatrixBinaryExpr<ML, MR>::Scalar;

		constexpr MatrixMultiplication(const ML& lhs, const MR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		static constexpr size_t Width() noexcept { return MR::Width(); }
		static constexpr size_t Height() noexcept { return ML::Height(); }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			Scalar result = 0;
			for (std::size_t i = 0; i < ML::Width(); ++i)
			{
				result += l_(row, i) * r_(i, column);
			}
			return result;
		}
	};


	/*--------------------------------------------------------------------
		Matrix Multiplication Operator: m * m
	--------------------------------------------------------------------*/
	template<Matrix ML, Matrix MR>
	inline constexpr auto operator*(const ML& lhs, const MR& rhs) noexcept -> MatrixMultiplication<ML, MR>
	{
		return MatrixMultiplication(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		2x2 Matrix Determinant Function: Det(m)
	--------------------------------------------------------------------*/
	template<Matrix M>
		requires MatrixOfKnownSize<M, 2, 2>
	inline constexpr auto Det(const M& m) -> typename M::Scalar
	{
		return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
	}


	/*--------------------------------------------------------------------
		3x3 Matrix Determinant Function: Det(m)
	--------------------------------------------------------------------*/
	template<Matrix M>
		requires MatrixOfKnownSize<M, 3, 3>
	inline constexpr auto Det(const M& m) -> typename M::Scalar
	{
		return
			  m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2,1))
			- m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
			+ m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
	}


	/*--------------------------------------------------------------------
		4x4 Matrix Determinant Function: Det(m)
	--------------------------------------------------------------------*/
	template<Matrix M>
		requires MatrixOfKnownSize<M, 4, 4>
	inline constexpr auto Det(const M& m) -> typename M::Scalar
	{
		return
			m(0,0) *
			(
				m(1, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2)) -
				m(2, 1) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2)) +
				m(3, 1) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2))
			) -
			m(1, 0) *
			(
				m(0, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2)) -
				m(2, 1) * (m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2)) +
				m(3, 1) * (m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2))
			) +
			m(2, 0) *
			(
				m(0, 1) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2)) -
				m(1, 1) * (m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2)) +
				m(3, 1) * (m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2))
			) -
			m(3, 0) *
			(
				m(0, 1) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2)) -
				m(1, 1) * (m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2)) +
				m(2, 1) * (m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2))
			);
	}
}


//========================================================================
#endif
