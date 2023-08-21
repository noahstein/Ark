/*************************************************************************
 * @file
 * @brief Vector concept and generic class implementation
 * 
 * @details Defines the Vector concept to represent the abstract 
 * definition of the structure of a mathematical vector. From there, this 
 * file implements many common vector functions on that abstraction. Due
 * to this implementation to the abstraction, each new concrete vector 
 * class oinly needs to define the type of its scalars and accessors to 
 * its elements, and it will not only automatically have a full set of 
 * operations, it will interact with any other mix of vector classes, as 
 * long as the scalar types may interact arithmetically.
 * 
 * Additionally, any specific vector class may define its own operators 
 * and implement optimized code for those oeprations. Classes may 
 * implement any operators they so choose and still rely upon the 
 * abstract implementations for the rest.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_VECTOR_H_INCLUDE_GUARD)
#define ARK_MATH_VECTOR_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <type_traits>

#include "Concepts.h"


//************************************************************************
//  Code
//************************************************************************
namespace ark::math
{
	//====================================================================
	//  Concepts
	//====================================================================

	/*********************************************************************
	 * @brief Vector Concept

	 * @details The Vector concept defines an abstract view of a vector.
	 * Please note this concept is a subset of the mathematical
	 * definition of a vector. The mathematical definition includes the
	 * basic arithmetic operations: addition, subtraction,
	 * multiplication by a scalar, etc. Those are no defined here. This
	 * concept models the definition of a vector as an ordered array of 
	 * numbers. Even the definitional arithmetic operations are 
	 * implemented against this definition thus providing maximum reuse 
	 * and minimizing the amount of unnecessary duplicated code.
	 *
	 * As such, a class models a vector if it meets 3 criteria:
	 *
	 * -# `Scalar`: The class must define this memeber type alias to 
	 *    indicate the type of all the vector's components. `Scalar` must 
	 *    be an arithmetic type as the mathematical concept of a vector 
	 *    defines addition and scalar multiplication of its components.
	 * -# `Size()`: The class mus define this member function to indicate 
	 *    its imensionality, that is, its number of components.
	 * -# `operator()`: The class must define this member function to 
	 *    return the values of its individual components.
	 * 
	 * @sa @ref ark::math::Arithmetic
	 * @sa [_Vector_ on Wolfram MathWorld]
	 *     (https://mathworld.wolfram.com/Vector.html)
	 * @sa [_Vector (mathematics and physics)_ on Wikipedia]
	 *     (https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics))
	 ********************************************************************/
	template<typename V>
	concept Vector = requires(V v)
	{
		typename V::Scalar;
		Arithmetic<typename V::Scalar>;

		{ v.Size() } -> std::same_as<std::size_t>;
		{ v(1) } -> std::same_as<typename V::Scalar>;
	};


	/*********************************************************************
	 * @brief Concept of Vectors Having the Same Dimension
	 * 
	 * @details Most binary vector arithmetic operations are defined
	 * only when both vectors are of the same dimension. This concept
	 * requires two classes be of the same dimension.
	 ********************************************************************/
	template<typename VL, typename VR>
	concept SameDimension = requires
	{
		requires (VL::Size() == VR::Size());
	};


	/*********************************************************************
	 * @brief Concept of a Vector of a Known Dimension
	 * 
	 * @details Model a vector of specific, known dimension. For example,
	 * the cross product is only defined for 2- and 3-dimensionalvectors, 
	 * and the the different dimensions have different semantics. This 
	 * concept provides a simple means of restricting and differentiating 
	 * the two functions.
	 ********************************************************************/
	template<typename V, int N>
	concept OfKnownDimension = requires (V v)
	{
		requires (V::Size() == N);
	};


	//====================================================================
	//  Expression Base Classes
	//====================================================================

	/*********************************************************************
	 * @brief Vector Expression
	 * 
	 * @details The base class for Vector-valued expressions. Before C++
	 * 20, this would be a template class using the CRTP pattern in 
	 * which the derived class type is passed to the base class and used 
	 * subsequently. Using concepts, it doesn't appear necessary; 
	 * however, if it proves so, this could change in the future. 
	 * Presently, the class provides a common base which can prove useful 
	 * in other concepts and as a source of common code for other classes 
	 * participating in quaternion expressions.
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	class VectorExpr
	{
	};


	/*********************************************************************
	 * @brief Vector Unary Expression Base Class
	 * 
	 * @tparam V The class of the unary vector expresion which must 
	 * fulfill the requirements of the @ref ark::math::Vector concept to 
	 * successfully compile.
	 * 
	 * @details The base class for vector-valued expressions with a 
	 * single operand. As the current implementation does not 
	 * functionally rely upon CRTP, this base class isn't strictly 
	 * necessary. Unary vector expression classes shall derive from 
	 * this base class to ensure proper classification in concept and 
	 * template expressions that classify nodes as unary vector 
	 * expressions. 
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<class V>
	class VectorUnaryExpr : public VectorExpr
	{
	public:
		/**
		 * @brief The type of the individual scalar components
		 * @details As a unary operation, the type of the resulting 
		 * scalar components is usually the same as the sole operand's 
		 * component scalar type.
		 */
		using Scalar = typename V::Scalar;
	};


	/*********************************************************************
	 * @brief Vector Binary Expression Base Class
	 * 
	 * @tparam VL The class of the vector expression node on the left side
	 * side of the expression. This must fulfill the requirements of the 
	 * @ref ark::math::Vector concept to be valid.
	 * 
	 * @tparam VR The class of the vector expression node on the 
	 * right side of the expression. This must fulfill the requirements 
	 * of the @ref ark::math::Vector concept to be valid.
	 * 
	 * @details This is the base class for all vector expression node 
	 *  clases with two operands. It provides easy access to the 
	 * appropriate `Scalar` type given the types in the two operands' 
	 * classes. Additionally, the class includes a constraint to ensure 
	 * that the two `Scalar` types may partake in mathematical 
	 * expressions and produce results. 
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires MutuallyArithmetic<typename VL::Scalar, typename VR::Scalar>
		&& SameDimension<VL, VR>
	class VectorBinaryExpr : public VectorExpr
	{
		/**
		 * @brief The scalar type of the left-hand vector expression
		 */
		using SL = typename VL::Scalar;

		/**
		 * @brief The scalar type of the right-hand vector expression
		 */
		using SR = typename VR::Scalar;

	public:
		/**
		 * @brief The type of the individual scalar components.
		 * 
		 * @details As a binary operation, the type of the resulting 
		 * scalar components is usually the type that results from 
		 * the usual promotion rules when using variables of those 
		 * types together in an arithmetic expression.
		 */
		using Scalar = typename std::common_type<SL, SR>::type;

		/**
		 * @brief The dimension of the vector
		 * 
		 * @details The dimension of the vector is the number of 
		 * components present in the vector.
		 * 
		 * @return constexpr std::size_t Number of components
		 */
		static constexpr std::size_t Size() noexcept { return VL::Size(); }
	};


	//====================================================================
	//  Expression Function Classes
	//====================================================================

	/*********************************************************************
	 * @brief Vector Negation Expression Node
	 * 
	 * @tparam V The class of the vector expression whose value is to 
	 * be negated by this node.
	 * This generally will not be specifired explicitly but will be 
	 * inferred by the compiler at the constructor call site.
	 * @sa @ref TemplateExpressionsAsClassParameters
	 * 
	 * @details @include{doc} Math/Vector/Negation.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief Vector Negation Operator:
	 *        \f$-\mathbf{v}\f$
	 * 
	 * @tparam V The class of the vector expression whose value is to 
	 * be negated by the constructed VectorNegation expression node.
	 * This generally will not be specifired explicitly but will be 
	 * inferred by the compiler at the operator call site.
	 * @sa @ref TemplateExpressionsAsClassParameters
	 * 
	 * @param v A vector expression whose value is to be negated.
	 * 
	 * @details This unary `operator-` constructs a negation expression 
	 * node around the vector expression argument.
	 * 
	 * @include{doc} Math/Vector/Negation.txt
	 * 
	 * @sa ark::math::VectorNegation
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector V>
	inline constexpr auto operator-(const V& v) noexcept -> VectorNegation<V>
	{
		return VectorNegation(v);
	}


	 /*********************************************************************
	 * @brief Vector Addition Expression Node
	 * 
	 * @tparam VL The class of the vector expression that is the addend 
	 * on the left-hand side of the plus.
	 * 
	 * @tparam VR The class of the vector expression that is the addend 
	 * on the right-hand side of the plus.
	 * 
	 * @details This node computes the sum of the two vector addends.
	 * 
	 * @include{doc} Math/Vector/Addition.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
	class VectorAddition : public VectorBinaryExpr<VL, VR>
	{
		using Expr = VectorBinaryExpr<VL, VR>;

		VL const & l_;
		VR const & r_;

	public:
		using Scalar = typename Expr::Scalar;

		constexpr VectorAddition(const VL& lhs, const VR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		constexpr Scalar operator()(std::size_t index) const { return l_(index) + r_(index); }
	};


	/*********************************************************************
	 * @brief Vector Addition Operator:
	 *       \f$\mathbf{v_1}+\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * addend on the left-hand side of the plus.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * addend on the right-hand side of the plus.
	 * 
	 * @param lhs The vector expression to the left of the +
	 * 
	 * @param rhs The vector expression to the right of the +
	 * 
	 * @details The binary `operator+` constructs a VectorAddition 
	 * instance with the two addends whose sum will be lazily computed at 
	 * a later time.
	 * 
	 * @include{doc} Math/Vector/Addition.txt
	 * 
	 * @sa ark::math::VectorAddition
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
	inline constexpr auto operator+(const VL& lhs, const VR& rhs) noexcept -> VectorAddition<VL, VR>
	{
		return VectorAddition(lhs, rhs);
	}


	 /*********************************************************************
	 * @brief Vector Subtraction Expression Node
	 * 
	 * @tparam VL The class of the vector expression that is the minuend 
	 * on the left-hand side of the plus.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * subtrahend on the right-hand side of the plus.
	 * 
	 * @details This node computes the difference of the two vectors.
	 * 
	 * @include{doc} Math/Vector/Subtraction.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief Vector Subtraction Operator:
	 *       \f$\mathbf{v_1}-\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * minuend on the left-hand side of the plus.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * subtrahend on the right-hand side of the plus.
	 * 
	 * @param lhs The mineund vector expression to the left of the +
	 * 
	 * @param rhs The subtrahend vector expression to the right of the +
	 * 
	 * @details The binary `operator-` constructs a VectorSubtraction 
	 * instance with the two vectors whose sum will be lazily computed at 
	 * a later time.
	 * 
	 * @include{doc} Math/Vector/Subtraction.txt
	 * 
	 * @sa ark::math::VectorSubtraction
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
	inline constexpr auto operator-(const VL& lhs, const VR& rhs) noexcept -> VectorSubtraction<VL, VR>
	{
		return VectorSubtraction(lhs, rhs);
	}


	 /*********************************************************************
	 * @brief Vector-Scalar Multplication Expression Node:
	 * \f$s\mathbf{v},\mathbf{v}s\f$
	 * 
	 * @tparam V The class of the quaternion expression to be multiplied 
	 * by a scalar.
	 * 
	 * @details This node computes the multpilication of a vector by a 
	 * scalar quantity. This class handles both forms \f$s\mathbf{v}\f$ 
	 * and \f$\mathbf{v}s\f$. Given the scalar \f$s\f$ and the vector 
	 * expression \f$\mathbf{v}\in\mathbb{R}^n=(v_0,...v_n)\f$, the 
	 * resulting vector is: 
	 * \f[\mathbf{v}s=(sv_0,...,sv_n)\f]
	 * or:
	 * \f[s\mathbf{v}=(sv_0,...,sv_n)\f]
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief Scalar-Vector Multiplication Operator:
	 *       \f$s*\mathbf{v}\f$
	 * 
	 * @tparam V The class of the vector expression that is the 
	 * multiplicand on the right-hand side of the *.
	 * 
	 * @param s The scalar multiplier on the left-hand side of the *.
	 * 
	 * @param v The vector expression multiplicand on the right- 
	 * hand side of the *.
	 * 
	 * @details The binary `operator*` constructs a 
	 * VectorScalarMultiplication instance with the multiplier and 
	 * multiplicand resulting in their product when the accessors are 
	 * called.
	 * 
	 * @include{doc} Math/Vector/ScalarVectorMultiplication.txt
	 * 
	 * @sa ark::math::VectorScalarMultiplication
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<typename S, Vector V>
		requires MutuallyArithmetic<S, typename V::Scalar>
	inline constexpr auto operator*(const S & s, const V & v) noexcept -> VectorScalarMultiplication<V, S>
	{
		return VectorScalarMultiplication(s, v);
	}


	/*********************************************************************
	 * @brief Vector-Scalar Multiplication Operator:
	 *       \f$\mathbf{v}*s\f$
	 * 
	 * @tparam V The class of the vector expression that is the 
	 * multiplier on the left-hand side of the *.
	 * 
	 * @param v The vector expression multiplier on the left-hand 
	 * side of the *.
	 * 
	 * @param s The scalar multiplicand on the right-hand side of the *.
	 * 
	 * @return An instance of VectorScalarMultiplication whose 
	 * accessors return the component values of a vector 
	 * multiplied by a scalar.
	 * 
	 * @details The binary `operator*`creates a template expression tree 
	 * node for evaluating the multiplication of a vector by a 
	 * scalar value.
	 * 
	 * @include{doc} Math/Vector/VectorScalarMultiplication.txt
	 * 
	 * @sa ark::math::ScalarVectorMultiplication
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector V, typename S>
	inline constexpr auto operator*(const V & v, const S & s) noexcept -> VectorScalarMultiplication<V, S>
	{
		return VectorScalarMultiplication(s, v);
	}


	 /*********************************************************************
	 * @brief Vector-Scalar Division Expression Node:
	 * \f$\mathbf{v}/s\f$
	 * 
	 * @tparam V The class of the dividend vector expression to be 
	 * divided by a scalar quantity.
	 * 
	 * @details This node lazily computes the quotient of the value of the 
	 * vector expression node and scalar passed in its constructor.
	 * 
	 * @include{doc} Math/Vector/ScalarDivision.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief Vector-Scalar Division Operator:
	 *       \f$\mathbf{v}/s\f$
	 * 
	 * @tparam V The class of the vector expression that is the 
	 * dividend on the left-hand side of the /.
	 * 
	 * @param v The vector expression dividend on the left-hand 
	 * side of the /.
	 * 
	 * @param s The scalar divisor on the right-hand side of the /.
	 * 
	 * @return An instance of VectorScalarDivision whose value is 
	 * lazily computed when the accessors are called.
	 * 
	 * @details The binary `operator/`creates a template expression tree 
	 * node for evaluating the division of a vector by a scalar. This 
	 * isn't defined strictly in mathematics; however, it is a good 
	 * short-hand for multiplying a vector by the multiplicative inverse 
	 * of a scalar.
	 * 
	 * @include{doc} Math/Vector/ScalarDivision.txt
	 * 
	 * @sa ark::math::VectorScalarDivision
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector V, typename S>
		requires MutuallyArithmetic<typename V::Scalar, S>
	inline constexpr auto operator/(const V & v, const S & s) noexcept -> VectorScalarDivision<V, S>
	{
		return VectorScalarDivision(v, s);
	}


	/*********************************************************************
	 * @brief Vector Equality Comparison Operator:
	 *       \f$\mathbf{v_1}==\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is on the  
	 * left-hand side of the ==.
	 * 
	 * @tparam VR The class of the vector expression that is on the  
	 * right-hand side of the ==.
	 * 
	 * @param lhs The vector expression to the left of the --
	 * 
	 * @param rhs The vector expression to the right of the ==
	 * 
	 * @return `true` if the two vectors are equal, `false` if unequal.
	 * 
	 * @details The binary `operator==` does not construct an expression 
	 * node. As the comparison results in a boolean result indicating 
	 * whether or not the two vectors are equal, the equality 
	 * comparison will not be a operand to a parent vector-valued 
	 * epxression. Therefore, nothing is constructed, the results of 
	 * the comparison are returned directly.
	 * 
	 * @include{doc} Math/Vector/Equality.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires SameDimension<VL, VR>
	inline constexpr auto operator==(const VL& lhs, const VR& rhs) -> bool
	{
		using Expr = VectorBinaryExpr<VL, VR>;
		std::size_t length = lhs.Size();
		for (std::size_t i = 0; i < length; ++i)
		{
			if (lhs(i) != rhs(i))
			{
				return false;
			}
		}
		return true;
	}


	/*********************************************************************
	 * @brief Vector Dot Product Function:
	 *        \f$\mathbf{v_1}\cdot\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the dot product.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the dot product.
	 * 
	 * @param lhs The vector expression that is the first factor.
	 * 
	 * @param rhs The vector expression that is the second factor.
	 * 
	 * @return The value of the dot product.
	 * 
	 * @details @include{doc} Math/Vector/DotProduct.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires SameDimension<VL, VR>
	inline constexpr auto Dot(const VL& l, const VR& r) -> typename VectorBinaryExpr<VL, VR>::Scalar
	{
		using Expr = VectorBinaryExpr<VL, VR>;
		typename Expr::Scalar result{0};
		std::size_t length = l.Size();
		for (std::size_t i = 0; i < length; ++i)
		{
			result += l(i) * r(i);
		}
		return result;
	}


	/*********************************************************************
	 * @brief Vector Norm Function:
	 *        \f$\mathrm{Norm}(\mathbf{v})\f$
	 * 
	 * @tparam V The class of the vector expression whose norm is to be 
	 * taken.
	 * 
	 * @param v The vector expression whose norm is to be computed.
	 * 
	 * @return The norm of the vector passed in as an argument.
	 * 
	 * @details Compute the norm of a given vector. This is the 
	 * Euclidean distance of the vector from the origin. It is also 
	 * called magnitude or length.
	 * 
	 * @include{doc} Math/Vector/Norm.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector V>
	inline constexpr auto Norm(const V& v) -> typename VectorUnaryExpr<V>::Scalar
	{
		return std::sqrt(Dot(v, v));
	}


	/*********************************************************************
	 * @brief 2-D Vector Cross Product Function:
	 *        \f$\mathbf{v_1}\times\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the cross product, on the left-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the cross product, on the right-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @param lhs The vector expression that is the first factor.
	 * 
	 * @param rhs The vector expression that is the second factor.
	 * 
	 * @return The scalar value of the 2-D cross product.
	 * 
	 * @details @include{doc} Math/Vector/CrossProduct2D.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 2> && OfKnownDimension<VR, 2>
	inline constexpr auto Cross(const VL& lhs, const VR& rhs) -> typename VectorBinaryExpr<VL, VR>::Scalar
	{
		return lhs(0) * rhs(1) - lhs(1) * rhs(0);
	}


	/*********************************************************************
	 * @brief 3-D Vector Cross Product Expression Node:
	 *        \f$\mathbf{v_1}\times\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the cross product, on the left-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the cross product, on the right-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @details @include{doc} Math/Vector/CrossProduct3D.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
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


	/*********************************************************************
	 * @brief 3-D Vector Cross Product Function:
	 *        \f$\mathbf{v_1}\times\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the cross product, on the left-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the cross product, on the right-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @param lhs The vector expression that is the first factor.
	 * 
	 * @param rhs The vector expression that is the second factor.
	 * 
	 * @return The vector value of the 3-D cross product.
	 * 
	 * @details @include{doc} Math/Vector/CrossProduct3D.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 3> && OfKnownDimension<VR, 3>
	inline constexpr auto Cross(const VL& lhs, const VR& rhs) noexcept -> VectorCrossProduct3D<VL, VR>
	{
		return VectorCrossProduct3D(lhs, rhs);
	}


	/*********************************************************************
	 * @brief 4-D Vector Cross Product Expression Node:
	 *        \f$\mathbf{v_1}\times\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the cross product, on the left-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the cross product, on the right-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @details Mathematically, there is no 4-D cross product. It is 
	 * defined here because it is very common to use 4-D vectors in 
	 * interactive 3D applications. As such, this 4-D cross product is 
	 * actually a 3-D cross product excluding the last elements of the 
	 * vectors.
	 * 
	 * @include{doc} Math/Vector/CrossProduct4D.txt
	 * 
	 * @sa ark::math::Vector
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 4> && OfKnownDimension<VR, 4>
	class VectorCrossProduct4D : public VectorBinaryExpr<VL, VR>
	{
		using Expr = VectorBinaryExpr<VL, VR>;

	public:
		using Scalar = typename Expr::Scalar;

	private:
		VL const l_;
		VR const r_;

		Scalar result[4];

	public:
		constexpr VectorCrossProduct4D(const VL& lhs, const VR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{
			result[0] = lhs(1) * rhs(2) - lhs(2) * rhs(1);
			result[1] = lhs(2) * rhs(0) - lhs(0) * rhs(2);
			result[2] = lhs(0) * rhs(1) - lhs(1) * rhs(0);
			result[3] = static_cast<Scalar>(0);
		}

		static constexpr std::size_t Size() noexcept { return 4; }
		constexpr Scalar operator()(std::size_t index) const { return result[index]; }
	};


	/*********************************************************************
	 * @brief 4-D Vector Cross Product Function:
	 *        \f$\mathbf{v_1}\times\mathbf{v_2}\f$
	 * 
	 * @tparam VL The class of the vector expression that is the 
	 * first factor of the cross product, on the left-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @tparam VR The class of the vector expression that is the 
	 * second factor of the cross product, on the right-hand side of the 
	 * \f$\times\f$.
	 * 
	 * @param lhs The vector expression that is the first factor.
	 * 
	 * @param rhs The vector expression that is the second factor.
	 * 
	 * @return The vector value of the 3-D cross product of the first 
	 * 3 components of the 4-D vectors.
	 * 
	 * @details @include{doc} Math/Vector/CrossProduct4D.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Vector VL, Vector VR>
		requires OfKnownDimension<VL, 4> && OfKnownDimension<VR, 4>
	inline constexpr auto Cross(const VL& lhs, const VR& rhs) noexcept -> VectorCrossProduct4D<VL, VR>
	{
		return VectorCrossProduct4D(lhs, rhs);
	}
}


//========================================================================
#endif
