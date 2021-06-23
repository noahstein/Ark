/*========================================================================
Description
	Defines the Vector concept to represent the abstract definition
	of the structure of a mathematical vector. From there, this file
	implements many common vector functions on that abstraction. Due
	to this implementation to the abstraction, each new concrete
	vector class oinly needs to define the type of its scalars and
	accessors to its elements, and it will not only automatically have
	a full set of operations, it will interact with any other mix of
	vector classes, as long as the scalar types may interact
	arithmetically.

	Additionally, any specific vector class may define its own
	operators and implement optimized code for those oeprations. Classes
	may implement any operators they so choose and still rely upon the
	abstract implementations for the rest.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#pragma once


/*========================================================================
	Dependencies
========================================================================*/
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
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
		@brief Vector Concept

		@details The Vector concept defines an abstract view of a vector.
		Please note this concept is a subset of the mathematical
		definition of a vector. The mathematical definition includes the
		basic arithmetic operations: addition, subtraction,
		multiplication by a scalar, etc. Those are no defined here. This
		concept models the definition of a vector as an array of numbers.
		Even the definitional arithmetic operations are implemented
		against this definition thus providing maximum reuse and
		minimizing the amount of unnecessary duplicated code.

		As such, a class models a vector if it meets 3 criteria:

		1) It defines a type Scalar to indicate the type of all its
		components. The Scalar must be an arithmetic type as the
		mathematical concept defines addition and multiplication of
		it components.

		2) It defines a static member function Size() to indicate its
		dimensionality, that is, it's number of components.

		3) It has an operator() to return the values of its components.
	--------------------------------------------------------------------*/
	template<typename V>
	concept Vector = requires(V v)
	{
		typename V::Scalar;
		Arithmetic<typename V::Scalar>;

		{ v.Size() } -> std::same_as<std::size_t>;
		{ v(1) } -> std::same_as<typename V::Scalar>;
	};


	/*--------------------------------------------------------------------
		@brief SameDimension Concept

		@details Most binary vector arithmetic operations are defined
		only when both vectors are of the same dimension. This concept
		requires two classes be of the saem dimension.
	--------------------------------------------------------------------*/
	template<typename VL, typename VR>
	concept SameDimension = requires
	{
		requires (VL::Size() == VR::Size());
	};


	/*--------------------------------------------------------------------
		@brief OfKnownDimension Concept

		@details Model a vector of specific, known dimension. For example,
		the cross product is only defined for 2- and 3-dimensional
		vectors, and the the different dimensions have different
		semantics. This concept provides a simple means of restricting
		and differentiating the two functions.
	--------------------------------------------------------------------*/
	template<typename V, int N>
	concept OfKnownDimension = requires (V v)
	{
		requires (V::Size() == N);
	};


	/*====================================================================
		Expression Base Classes
	====================================================================*/

	/*--------------------------------------------------------------------
		Base class of all vector expressions
	--------------------------------------------------------------------*/
	class VectorExpr
	{
	};


	/*--------------------------------------------------------------------
		Base class of 1-argument vector expressions
	--------------------------------------------------------------------*/
	template<class V>
	class VectorUnaryExpr : public VectorExpr
	{
	public:
		using Scalar = V::Scalar;
	};


	/*--------------------------------------------------------------------
		Base class of 2-argument vector expressions
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires MutuallyArithmetic<typename VL::Scalar, typename VR::Scalar>
		&& SameDimension<VL, VR>
	class VectorBinaryExpr : public VectorExpr
	{
		using SL = typename VL::Scalar;
		using SR = typename VR::Scalar;

	public:
		using Scalar = typename std::common_type<SL, SR>::type;

		static constexpr std::size_t Size() noexcept { return VL::Size(); }
		static constexpr auto Range()
		{
			return std::views::iota(std::size_t{ 0 }, Size());
		}
};


	/*====================================================================
		Expression Function Classes
	====================================================================*/

	/*--------------------------------------------------------------------
		Vector Negation Expression: -v
	--------------------------------------------------------------------*/
	template<Vector V>
	class VectorNegation : public VectorUnaryExpr<V>
	{
		V const & v_;

	public:
		using Scalar = typename VectorUnaryExpr<V>::Scalar;

		constexpr VectorNegation(const V& v) noexcept
			: v_(v)
		{}

		static constexpr size_t Size() noexcept { return V::Size(); }
		constexpr Scalar operator()(size_t index) const
		{
			return -v_(index);
		}
	};


	/*----------------------------------------------------------------
		Vector Negation Operator: -v
	----------------------------------------------------------------*/
	template<Vector V>
	inline constexpr auto operator-(const V& v) noexcept -> VectorNegation<V>
	{
		return VectorNegation(v);
	}


	/*--------------------------------------------------------------------
		Vector Addition Expression: v1 + v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
	class VectorAddition : public VectorBinaryExpr<VL, VR>
	{
		using Expr = typename VectorBinaryExpr<VL, VR>;

		VL const & l_;
		VR const & r_;

	public:
		using Scalar = Expr::Scalar;

		constexpr VectorAddition(const VL& lhs, const VR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		constexpr Scalar operator()(std::size_t index) const { return l_(index) + r_(index); }
	};


	/*--------------------------------------------------------------------
		Vector Addition Operator: v1 + v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
	inline constexpr auto operator+(const VL& lhs, const VR& rhs) noexcept -> VectorAddition<VL, VR>
	{
		return VectorAddition(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		Vector Subtraction Expression: v1 - v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
	class VectorSubtraction : public VectorBinaryExpr<VL, VR>
	{
		VR const & r_;
		VL const & l_;

		using Expr = VectorBinaryExpr<VL, VR>;

	public:
		using Scalar = typename Expr::Scalar;

		constexpr VectorSubtraction(const VL& lhs, const VR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		static constexpr std::size_t Size() noexcept { return Expr::Size(); }
		constexpr Scalar operator()(std::size_t index) const { return l_(index) - r_(index); }
	};


	/*--------------------------------------------------------------------
		Vector Subtraction Operator: v1 - v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
	inline constexpr auto operator-(const VL& lhs, const VR& rhs) noexcept -> VectorSubtraction<VL, VR>
	{
		return VectorSubtraction(lhs, rhs);
	}


	/*--------------------------------------------------------------------
		Vector-Scalar Multiplication Expression: s * v, v * s
	--------------------------------------------------------------------*/
	template<Vector V, typename S>
		requires MutuallyArithmetic<typename V::Scalar, S>
	class VectorScalarMultiplication : public VectorExpr
	{
	public:
		using Scalar = S;

	private:
		Scalar const & s_;
		V const & v_;

	public:
		constexpr VectorScalarMultiplication(const Scalar & s, const V & v) noexcept
			: s_(s), v_(v)
		{}

		static constexpr std::size_t Size() noexcept { return V::Size(); }
		constexpr Scalar operator()(std::size_t index) const { return v_(index) * s_; }
	};


	/*--------------------------------------------------------------------
		Scalar-Vector Multiplication Operator: s * v
	--------------------------------------------------------------------*/
	template<typename S, Vector V>
		requires MutuallyArithmetic<S, typename V::Scalar>
	inline constexpr auto operator*(const S & s, const V & v) noexcept -> VectorScalarMultiplication<V, S>
	{
		return VectorScalarMultiplication(s, v);
	}


	/*--------------------------------------------------------------------
		Vector-Scalar Multiplication Operator: v * s
	--------------------------------------------------------------------*/
	template<Vector V, typename S>
	inline constexpr auto operator*(const V & v, const S & s) noexcept -> VectorScalarMultiplication<V, S>
	{
		return VectorScalarMultiplication(s, v);
	}


	/*--------------------------------------------------------------------
		Vector-Scalar Division Expression: v / s
	--------------------------------------------------------------------*/
	template<Vector V, typename S>
		requires MutuallyArithmetic<typename V::Scalar, S>
	class VectorScalarDivision : public VectorExpr
	{
	public:
		using Scalar = S;

	private:
		V const & v_;
		Scalar const & s_;

	public:
		constexpr VectorScalarDivision(const V& v, const Scalar & s) noexcept
			: v_(v), s_(s)
		{}

		static constexpr std::size_t Size() noexcept { return V::Size(); }
		constexpr Scalar operator()(std::size_t index) const { return v_(index) / s_; }
	};


	/*--------------------------------------------------------------------
		Vector-Scalar Division Operator: v / s
	--------------------------------------------------------------------*/
	template<Vector V, typename S>
		requires MutuallyArithmetic<typename V::Scalar, S>
	inline constexpr auto operator/(const V & v, const S & s) noexcept -> VectorScalarDivision<V, S>
	{
		return VectorScalarDivision(v, s);
	}


	/*--------------------------------------------------------------------
		Vector Equality Operator: v1 == v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires SameDimension<VL, VR>
	inline constexpr auto operator==(const VL& lhs, const VR& rhs) -> bool
	{
		using Expr = VectorBinaryExpr<VL, VR>;
		bool result = std::ranges::all_of(Expr::Range(), [&](std::size_t i) { return lhs(i) == rhs(i); });
		return result;
	}


	/*--------------------------------------------------------------------
		Dot Product Function: Dot(v)
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires SameDimension<VL, VR>
	inline constexpr auto Dot(const VL& l, const VR& r) -> typename VectorBinaryExpr<VL, VR>::Scalar
	{
		using Expr = VectorBinaryExpr<VL, VR>;
		typename Expr::Scalar result{0};
		std::ranges::for_each(Expr::Range(), [&](std::size_t i) { result += l(i) * r(i); });
		return result;
	}


	/*--------------------------------------------------------------------
		Magnitude Function: Magnitude(v)
	--------------------------------------------------------------------*/
	template<Vector V>
	inline constexpr auto Magnitude(const V& v) -> typename VectorUnaryExpr<V>::Scalar
	{
		return std::sqrt(Dot(v, v));
	}


	/*--------------------------------------------------------------------
		2D Cross Product Function: Cross(v)
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 2> && OfKnownDimension<VR, 2>
	inline constexpr auto Cross(const VL& lhs, const VR& rhs) -> typename VectorBinaryExpr<VL, VR>::Scalar
	{
		return lhs(0) * rhs(1) - lhs(1) * rhs(0);
	}


	/*--------------------------------------------------------------------
		3D Cross Product Expression: v1 x v2
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 3> && OfKnownDimension<VR, 3>
	class VectorCrossProduct3D : public VectorBinaryExpr<VL, VR>
	{
		using Expr = VectorBinaryExpr<VL, VR>;

	public:
		using Scalar = typename Expr::Scalar;

	private:
		VL const l_;
		VR const r_;

		Scalar result[3];

	public:
		constexpr VectorCrossProduct3D(const VL& lhs, const VR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{
			result[0] = lhs(1) * rhs(2) - lhs(2) * rhs(1);
			result[1] = lhs(2) * rhs(0) - lhs(0) * rhs(2);
			result[2] = lhs(0) * rhs(1) - lhs(1) * rhs(0);
		}

		static constexpr std::size_t Size() noexcept { return 3; }
		constexpr Scalar operator()(std::size_t index) const { return result[index]; }
	};


	/*--------------------------------------------------------------------
		3D Cross Product Function: Cross(v1, v2)
	--------------------------------------------------------------------*/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 3> && OfKnownDimension<VR, 3>
	inline constexpr auto Cross(const VL& lhs, const VR& rhs) noexcept -> VectorCrossProduct3D<VL, VR>
	{
		return VectorCrossProduct3D(lhs, rhs);
	}
}
