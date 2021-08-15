/*************************************************************************
 * @file
 * @brief Quaternion concept and generic class implementation
 * 
 * @details This file defines a C++ 20 concept of a mathematical 
 * Quaternion. It also contains a generic expression template 
 * implementation of Quaternion written against the abstract concept. Thus, 
 * the implementation here is, itself, abstract and is insufficient on its 
 * own to be used in code. Projects may have more than one concrete 
 * quaternion class. By coding to the Quaternion abstraction, each 
 * concrete class will interoperate out of the box with each other as long 
 * as the scalar numeric types interact. See the @ref Quat template class 
 * for a simple, yet extensible, realization of the Quaternion concept.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

#if !defined(ARK_MATH_QUATERNION_H_INCLUDE_GUARD)
#define ARK_MATH_QUATERNION_H_INCLUDE_GUARD


//************************************************************************
//  Dependencies
//************************************************************************
#include <cmath>
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
	 * @brief Quaternion Concept Defining 4D Complex Number Classes
	 * 
	 * @details This concept models an abstraction of the mathematical 
	 * entity of a quatnerion, a 4-dimensional complex number. A 
	 * quaternion is written in the form \f$(w, x, y, z)\f$ where each 
	 * component is a real number: \f$w\f$ is the _scalar_ component and 
	 * \f$x, y, z\f$ form the _vector_. The scalar is the real part of 
	 * the quaternion and the vector forms the complex part. It is 
	 * analogous to the form:
	 * \f[w+x\mathbf{i}+y\mathbf{j}+z\mathbf{k}\f]
	 * where \f$\mathbf{i},\mathbf{j},\mathbf{k}\f$ are the three axes 
	 * in the complex space. A quaternion of unit magnitude may be used 
	 * to model 3D rotation. In that situation, the three complex 
	 * components correspond to a vector defining the axis of rotation.
	 * 
	 * Mathematically, a quaternion is a non-abelian field, which means 
	 * it's a noncommutative ring with inverse. That itself means that 
	 * it's a set of numbers that's closed in regard to the operations 
	 * addition, subtraction, multiplication, and division and that 
	 * multiplication distributes over addition. These properties may 
	 * be summarized as follows. Given \f$\mathbb{Q}\f$ is the set of 
	 * quaternions and \f$\mathbf{a},\mathbf{b} \in \mathbb{Q}\f$.
	 * 
	 * For addition:
	 * 
	 * - Closure: \f$\mathbf{a}+\mathbf{b} \in \mathbb{Q}\f$
	 * - Associativity: \f$(\mathbf{a}+\mathbf{b})+\mathbf{c}
	 *                  =\mathbf{a}+(\mathbf{b}+\mathbf{c})\f$
	 * - Identity: \f$\mathbf{a}+\mathbf{0}
	 *             =\mathbf{0}+\mathbf{a}=\mathbf{a}\f$
	 * - Inverse: \f$\mathbf{a}+(-\mathbf{a})
	 *            =(-\mathbf{a})+\mathbf{a}=\mathbf{0}\f$
	 * 
	 * For multiplication:
	 * 
	 * - Closure: \f$\mathbf{ab} \in \mathbb{Q}\f$
	 * - Associativity: \f$(\mathbf{ab})\mathbf{c}
	 *                  =\mathbf{a}(\mathbf{bc})\f$
	 * - Identity: \f$a \times 1=1 \times a=a\f$
	 * - Inverse: \f$a a^{-1}=a^{-1} a=1\f$
	 * 
	 * For distribution of multiplicatin over addition:
	 * 
	 * - \f$\mathbf{a}(\mathbf{b}+\mathbf{c})
	 *   =\mathbf{ab}+\mathbf{ac}\f$
	 * - \f$(\mathbf{a}+\mathbf{b})\mathbf{c}
	 *   =\mathbf{ac}+\mathbf{bc}\f$
	 * 
	 * The concept cannot validate the aforementioned properties. The 
	 * concept ensures that any quaternion in the closs models the 
	 * structure of a quaternion's in code. It ensures any class 
	 * attmpeting to be used as a quaternion meets the following 
	 * structural rqeuirements:
	 * 
	 * - An embedded typedef defining the type of the components named 
	 *   `Scalar`. This type must support the standard algebraic 
	 *   operations returning a result of `Scalar` type or one that is 
	 *   convertible to that type.
	 * - Four accessor functions to read the individual components of the 
	 *   quaternion: `w()`, `x()`, `y()`, `z()`. The accessors returns 
	 *   values of the type the class defines as `Scalar`.
	 * 
	 * @note The accessors are functions, not data. This ensures that the 
	 * abstract concept places no rquirements on the data structure of 
	 * the quatarnion type or even if it uses per-instance data at all. 
	 * In optimized builds, accessors tend to be so simple that they 
	 * inline with no penalty when retrieving simple data layouts.
	 * 
	 * @note THe conceptt places no requirements on constructors. As it 
	 * cannot know the structure of the concrete implementations, it is 
	 * unwise to place limits. C++ requires constructors to be part of 
	 * the exact class being constructed, so any requirements can be 
	 * placed on the individual types themselves.
	 * 
	 * @sa @ref ark::math::Arithmetic
	 * @sa [_Quaternion_ on Wolfram MathWorld]
	 *     (https://mathworld.wolfram.com/Quaternion.html)
	 * @sa [_Quaternionon_ on Wikipedia]
	 *     (https://en.wikipedia.org/wiki/Quaternion)
	 * @sa [_Quaternions and spatial rotation_ on Wikipedia]
	 *     (https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
	 ********************************************************************/
	template<typename Q>
	concept Quaternion = requires(Q q)
	{
		typename Q::Scalar;
		Arithmetic<typename Q::Scalar>;

		{ q.w() } -> std::same_as<typename Q::Scalar>;
		{ q.x() } -> std::same_as<typename Q::Scalar>;
		{ q.y() } -> std::same_as<typename Q::Scalar>;
		{ q.z() } -> std::same_as<typename Q::Scalar>;
	};


	//====================================================================
	//  Expression Base Classes
	//====================================================================

	/*********************************************************************
	 * @brief Quaternion Expression
	 * 
	 * @details The base class for quaternion-valued expressions. Before 
	 * C++ 20, this would be a template class using the CRTP pattern in 
	 * which the derived class type is passed to the base class and used 
	 * subsequently. Using concepts, it doesn't appear necessary; 
	 * however, if it proves so, this could change in the future. 
	 * Presently, the class provides a common base which can prove useful 
	 * in other concepts and as a source of common code for other classes 
	 * participating in quaternion expressions.
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	class QuaternionExpr
	{
	protected:
		/*****************************************************************
		 * @brief Cached Quaternion Value
		 * 
		 * @tparam S The expression node's scalar type as determined from 
		 * the result of the operation applied to the operands.
		 *  
		 * @details Most concrete quaternion expression classes merely 
		 * access operands and combine them in simple ways, so caching 
		 * values is unnecessary. Some operations, e.g. multiplication, 
		 * do not lend themselves to such an approach. Each component 
		 * relies on a large, complex computation common to all the 
		 * components and would result in a tremendous duplication of 
		 * processing in debug builds and possibly even in release. In 
		 * those cases, the expression nodes will compute the value in 
		 * the constructor and then simply serve up the values in the 
		 * accessor functions.
		 * 
		 * @sa ark::math::Quaternion
		 ****************************************************************/
		template<typename Scalar>
		struct Cache
		{
			Scalar w, x, y, z;
		};
	};


	/*********************************************************************
	 * @brief Quaternion Unary Expression Base Class
	 * 
	 * @tparam Q The class of the unary quaternion expression which must 
	 * fulfill the reuquirements of the @ref ark::math::Quaternion 
	 * concept to successfully compile.
	 * 
	 * @details The base class for quaternion-valoued expressions with a 
	 * single operand. As the current implementation does not 
	 * functionally rely upon CRTP, this base class isn't strictly 
	 * necessary. Unary quaternion  expression classes shall derive from 
	 * his base class to ensure proper classification in concept and 
	 * template expressions that classify nodes as unary quaternion 
	 * expressions. 
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	class QuaternionUnaryExpr : public QuaternionExpr
	{
	public:
		/// @brief Convenince definition in the base class.
		using Scalar = typename Q::Scalar;
	};


	/*********************************************************************
	 * @brief Quaternion Binary Expression Base Class
	 * 
	 * @tparam QL The class of the quaternion expression node on the left 
	 * side of the expression. This must fulfill the requirements of the 
	 * @ref ark::math::Quaternion concept to be valid.
	 * 
	 * @tparam QR The class of the quaternion expression node on the 
	 * right side of the expression. This must fulfill the requirements 
	 * of the @ref ark::math::Quaternion concept to be valid.
	 * 
	 * @details This is the base class for all quaternion expression 
	 * node clases with two operands. It provides easy access to the 
	 * appropriate `Scalar` type given the types in the two operands' 
	 * classes. Additionally, the class includes a constraint to ensure 
	 * that the two `Scalar` types may partake in mathematicsl 
	 * expressions and produce results. 
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
		requires MutuallyArithmetic<typename QL::Scalar, typename QR::Scalar>
	class QuaternionBinaryExpr : public QuaternionExpr
	{
		/// The scalar type of the left-hand quaternion expression
		using SL = typename QL::Scalar;
		/// The scalar type of the right-hand quaternion expression
		using SR = typename QR::Scalar;

	public:
		/// The resulting type of using the two scalar types together
		using Scalar = typename std::common_type<SL, SR>::type;
	};


	//====================================================================
	//  Expression Function Classes
	//====================================================================

	/*********************************************************************
	 * @brief Quaternion Negation Expression Node
	 * 
	 * @tparam Q The class of the quaternion expression whose value is to 
	 * be negated by this node.
	 * This generally will not be specifired explicitly but will be 
	 * inferred by the compiler at the constructor call site.
	 * @sa @ref TemplateExpressionsAsClassParameters
	 * 
	 * @details @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	class QuaternionNegation : QuaternionUnaryExpr<Q>
	{
		/// Quaternion expression to negate
		Q q_;

	public:
		/// The negation has the same scalar type as the unary operand.
		using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

		/**
		 * @brief Quaternion expression constructor
		 * @param q Any arbitrary quaternion expression template tree.
		 * @details @include{doc} Math/Quaternion/UnaryParameterConceptDetails.txt
		 */
		QuaternionNegation(const Q& q) noexcept
			: q_(q)
		{}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return -q_.w(); }
		Scalar x() const noexcept { return -q_.x(); }
		Scalar y() const noexcept { return -q_.y(); }
		Scalar z() const noexcept { return -q_.z(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Negation Operator:
	 *        \f$-\mathbf{q}\f$
	 * 
	 * @tparam Q The class of the quaternion expression whose value is to 
	 * be negated by the constructed QuaternionNegation expression node.
	 * This generally will not be specifired explicitly but will be 
	 * inferred by the compiler at the operator call site.
	 * @sa @ref TemplateExpressionsAsClassParameters
	 * 
	 * @param q A quaternion expression whose value is to be negated.
	 * 
	 * @details This unary `operator-` constructs a negation expression 
	 * node around the quaternion expression argument.
	 * 
	 * @include{doc} Math/Quaternion/Negation.txt
	 * 
	 * @sa ark::math::QuaternionNegation
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto operator-(const Q& q) noexcept -> QuaternionNegation<Q>
	{
		return QuaternionNegation(q);
	}


	 /*********************************************************************
	 * @brief Quaternion Conjugation Expression Node
	 * 
	 * @tparam Q The class of the quaternion expression the value of 
	 * whose conjugate will be returned.
	 * 
	 * @details @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<typename Q>
	class QuaternionConjugation : QuaternionUnaryExpr<Q>
	{
		/// A quaternion to return the conjugate of.
		Q q_;

	public:
		/// The conjugate uses the same scalar type as the subexpression.
		using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion expression constructor
		 * @param q Any arbitrary quaternion expression template tree.
		 * @details @include{doc} Math/Quaternion/UnaryParameterConceptDetails.txt
		 */ 
		QuaternionConjugation(const Q& q) noexcept
			: q_(q)
		{}
		/// @}


		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return  q_.w(); }
		Scalar x() const noexcept { return -q_.x(); }
		Scalar y() const noexcept { return -q_.y(); }
		Scalar z() const noexcept { return -q_.z(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Conjugation Operator: 
	 *        \f$\mathbf{q}^*\f$
	 * 
	 * @tparam Q The class of the quaternion expression the conjugate of 
	 * whose value is to returned by the QuaternionNegation expression 
	 * node instance.
	 * 
	 * @param q A quaternion expression the conjugate of whose value is 
	 * to be returned.
	 * 
	 * @details This unary `operator*` constructs a conjugation expression 
	 * node around the quaternion expression argument.
	 * 
	 * @details @include{doc} Math/Quaternion/Conjugation.txt
	 * 
	 * @sa ark::math::QuaternionConjugate
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto operator*(const Q& q) noexcept -> QuaternionConjugation<Q>
	{
		return QuaternionConjugation(q);
	}


	 /*********************************************************************
	 * @brief Quaternion Addition Expression Node
	 * 
	 * @tparam QL The class of the quaternion expression that is the addend 
	 * on the left-hand side of the plus.
	 * 
	 * @tparam QR The class of the quaternion expression that is the addend 
	 * on the right-hand side of the plus.
	 * 
	 * @details This node computes the sum of the two quaternion addends.
	 * @include{doc} Math/Quaternion/UnaryParameterConceptDetails.txt
	 * 
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	class QuaternionAddition : public QuaternionBinaryExpr<QL, QR>
	{
		/// The quaternion expression addend on the left of the +
		QL l_;

		/// The quaternion expression addend on the right of the +
		QR r_;

	public:
		/// The resulting type of using the two scalar types together
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion expressions constructor
		 * 
		 * @param lhs The quaternion expression to the left of the +
		 * @param rhs The quaternion expression to the right of the +
		 */ 
		QuaternionAddition(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return l_.w() + r_.w(); }
		Scalar x() const noexcept { return l_.x() + r_.x(); }
		Scalar y() const noexcept { return l_.y() + r_.y(); }
		Scalar z() const noexcept { return l_.z() + r_.z(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Addition Operator:
	 *       \f$\mathbf{q_1}+\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * addend on the left-hand side of the plus.
	 * 
	 * @tparam QR The class of the quaternion expression that is the 
	 * addend on the right-hand side of the plus.
	 * 
	 * @param lhs The quaternion expression to the left of the +
	 * 
	 * @param rhs The quaternion expression to the right of the +
	 * 
	 * @details The binary `operator+` constructs a QuaternionAddition 
	 * isntance with the two addends whose sum will be lazily computed at 
	 * a later time.
	 * 
	 * @include{doc} Math/Quaternion/Addition.txt
	 * 
	 * @sa ark::math::QuaternionAddition
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto operator+(const QL& lhs, const QR& rhs) noexcept -> QuaternionAddition<QL, QR>
	{
		return QuaternionAddition(lhs, rhs);
	}


	 /*********************************************************************
	 * @brief Quaternion Subtraction Expression Node
	 * 
	 * @tparam QL The class of the minuend quaternion expression, the 
	 * number on the left-hand side of the minus.
	 * 
	 * @tparam QR The class of the subtrahend quaternion expression, the 
	 * number on the right-hand side of the plus.
	 * 
	 * @details This node computes the difference of the subtrahend 
	 * subtracted from the minuen. 
	 * 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	class QuaternionSubtraction : public QuaternionBinaryExpr<QL, QR>
	{
		/// The quaternion expression that is the minuend
		QL l_;

		/// The quaternion expression that is the subtrahed
		QR r_;

	public:
		/// The resulting type of using the two scalar types together
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion expressions constructor
		 * 
		 * @param lhs The quaternion expression to the left of the -
		 * @param rhs The quaternion expression to the right of the -
		 */ 
		QuaternionSubtraction(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return l_.w() - r_.w(); }
		Scalar x() const noexcept { return l_.x() - r_.x(); }
		Scalar y() const noexcept { return l_.y() - r_.y(); }
		Scalar z() const noexcept { return l_.z() - r_.z(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Subtraction Operator:
	 *       \f$\mathbf{q_1}-\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * minuend on the left-hand side of the minue.
	 * 
	 * @tparam QR The class of the quaternion expression that is the 
	 * subtrahend on the right-hand side of the minus.
	 * 
	 * @param lhs The quaternion expression to the left of the -
	 * 
	 * @param rhs The quaternion expression to the right of the -
	 * 
	 * @details The binary `operator-` constructs a QuaternionSubtraction 
	 * isntance with the minuend and subtrahend  resulting the their 
	 * difference when the accessors are called.
	 * 
	 * @include{doc} Math/Quaternion/Subtraction.txt
	 * 
	 * @sa ark::math::QuaternionSubtraction
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto operator-(const QL& lhs, const QR& rhs) noexcept -> QuaternionSubtraction<QL, QR>
	{
		return QuaternionSubtraction(lhs, rhs);
	}


	 /*********************************************************************
	 * @brief Quaternion-Scalar Multplication Expression Node:
	 * \f$s\mathbf{q},\mathbf{q}s\f$
	 * 
	 * @tparam Q The class of the quaternion expression to be multiplied 
	 * by a scalar.
	 * 
	 * @details This node computes the multpilication of a quaternion by a 
	 * scalar quantity. This class handles both forms \f$s\mathbf{q}\f$ 
	 * and \f$\mathbf{q}s\f$. Given the scalar \f$s\f$ and the quaternion 
	 * expression \f$\mathbf{q}=(w,x,y,z)\f$, the resulting quaternion is:
	 * \f[\mathbf{q}s=(sw,sx,sy,sz)\f]
	 * or:
	 * \f[s\mathbf{q}=(sw,sx,sy,sz)\f]
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	class QuaternionScalarMultiplication : public QuaternionExpr
	{
	public:
		using Scalar = typename Q::Scalar;

	private:
		/// The scalar factor in the scalar-quaternion multiplication
		Scalar s_;
		/// The quaternion factor in the scalar-quaternion multiplication
		Q q_;

	public:
		/// @name Constructors
		/// @{
		/**
		 * @brief Scalar-quaternion multiplication node epxression 
		 * constructor.
		 * 
		 * @param s The scalar multiplication factor
		 * @param q The quaternion expression factor
		 */ 
		QuaternionScalarMultiplication(Scalar s, const Q& q) noexcept
			: s_(s), q_(q)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return s_ * q_.w(); }
		Scalar x() const noexcept { return s_ * q_.x(); }
		Scalar y() const noexcept { return s_ * q_.y(); }
		Scalar z() const noexcept { return s_ * q_.z(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Scalar-Quaternion Multiplication Operator:
	 *       \f$s*\mathbf{q}\f$
	 * 
	 * @tparam Q The class of the quaternion expression that is the 
	 * multiplicand on the right-hand side of the *.
	 * 
	 * @param s The scalar multiplier on the left-hand side of the *.
	 * 
	 * @param q The quaternion expression multiplicand on the right- 
	 * hand side of the *.
	 * 
	 * @details The binary `operator*` constructs a 
	 * QuaternionScalarMultiplication instance with the multiplier and 
	 * multiplicand resulting in their product when the accessors are 
	 * called.
	 * 
	 * @include{doc} Math/Quaternion/ScalarQuaternionMultiplication.txt
	 * 
	 * @sa ark::math::QuaternionScalarMultiplication
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto operator*(typename Q::Scalar s, const Q& q) noexcept -> QuaternionScalarMultiplication<Q>
	{
		return QuaternionScalarMultiplication(s, q);
	}


	/*********************************************************************
	 * @brief Quaternion-Scalar Multiplication Operator:
	 *       \f$\mathbf{q}*s\f$
	 * 
	 * @tparam Q The class of the quaternion expression that is the 
	 * multiplier on the left-hand side of the *.
	 * 
	 * @param q The quternion expression multiplier on the left-hand 
	 * side of the *.
	 * 
	 * @param s The scalar multiplicand on the right-hand side of the *.
	 * 
	 * @return An instance of QuaternionScalarMultiplication whose 
	 * accessors return the component values of a quaternion 
	 * multiplied by a scalar.
	 * 
	 * @details The binary `operator*`creates a template expression tree 
	 * node for evaluating the multiplication of a quaternion by a 
	 * scalar value.
	 * 
	 * @include{doc} Math/Quaternion/QuaternionScalarMultiplication.txt
	 * 
	 * @sa ark::math::ScalarQuaternionMultiplication
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto operator*(const Q& q, typename Q::Scalar s) noexcept -> QuaternionScalarMultiplication<Q>
	{
		return QuaternionScalarMultiplication(s, q);
	}


	 /*********************************************************************
	 * @brief Quaternion-Scalar Division Expression Node:
	 * \f$\mathbf{q}/s\f$
	 * 
	 * @tparam Q The class of the quaternion expression to be divided by a 
	 * scalar quantity.
	 * 
	 * @details This node computes the conjugate of the value of the 
	 * quaternion expression node passed in its constructor.
	 * 
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	class QuaternionScalarDivision : public QuaternionUnaryExpr<Q>
	{
	public:
		using Scalar = typename Q::Scalar;

	private:
		/// The quaternion expression to be divided
		Q q_;
		/// The scalar divisor to dvide the quaternion
		Scalar s_;

	public:
		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion-scalar expression node constructor.
		 * 
		 * @param q The quaternion expression dividends
		 * @param s The scalar divisor
		 */ 
		QuaternionScalarDivision(const Q& q, Scalar s) noexcept
			: q_(q), s_(s)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return q_.w() / s_; }
		Scalar x() const noexcept { return q_.x() / s_; }
		Scalar y() const noexcept { return q_.y() / s_; }
		Scalar z() const noexcept { return q_.z() / s_; }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion-Scalar Division Operator:
	 *       \f$\mathbf{q}/s\f$
	 * 
	 * @tparam Q The class of the quaternion expression that is the 
	 * dividend on the left-hand side of the /.
	 * 
	 * @param q The quternion expression dividend on the left-hand 
	 * side of the /.
	 * 
	 * @param s The scalar divisor on the right-hand side of the /.
	 * 
	 * @return An instance of QuaternionScalarDivision whose value is 
	 * lazily computed when the accessors are called.
	 * 
	 * @details The binary `operator/`creates a template expression tree 
	 * node for evaluating the division of a quaternion by a scalar.
	 * 
	 * @include{doc} Math/Quaternion/ScalarDivision.txt
	 * 
	 * @sa ark::math::QuaternionScalarDivision
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto operator/(const Q& q, typename Q::Scalar s) noexcept -> QuaternionScalarDivision<Q>
	{

		return QuaternionScalarDivision(q, s);
	}
	

	/*********************************************************************
	 * @brief Quaternion Euqality Comparison Operator:
	 *       \f$\mathbf{q_1}==\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is on the  
	 * left-hand side of the ==.
	 * 
	 * @tparam QR The class of the quaternion expression that is on the  
	 * right-hand side of the ==.
	 * 
	 * @param lhs The quaternion expression to the left of the -
	 * 
	 * @param rhs The quaternion expression to the right of the -
	 * 
	 * @return `true` if the two quaternions are equal, `false` if unequal.
	 * 
	 * @details The binary `operator==` does not construct an expression 
	 * node. As the comparison results in a boolean result indicating 
	 * whether or not the two quaternions are equal, the equality 
	 * comparison will not be a operand to a parent quaternion-valued 
	 * epxression. Therefore, nothing is constructed, the results of 
	 * the comparison are returned directly.
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto operator==(const QL& lhs, const QR& rhs) noexcept -> bool
	{
		return
			lhs.w() == rhs.w() && 
			lhs.x() == rhs.x() &&
			lhs.y() == rhs.y() &&
			lhs.z() == rhs.z();
	}


	 /*********************************************************************
	 * @brief Quaternion Multiplication Expression Node:
	 * \f$\mathbf{qq}\f$
	 * 
	 * @tparam QL The class of the quaternion expression multiplier that 
	 * is the factor on the left-hand side.
	 * 
	 * @param QR The clas of hte quaternion expression multiplicand that 
	 * is the factor on the right-hand side.
	 * 
	 * @details This node computes the product of multiplying together 
	 * two quaternions. Unlike most quaternion expression nodes, this one 
	 * evaluates the result early and caches the data for efficiency
	 * purposes; otherwise, each accessor would be required to compute 
	 * the value separately, which is 16 multiplies and 12 adds, a 
	 * significant amount of redundant computation as it is most likely 
	 * all four accessor will be called at least once.
	 * 
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	class QuaternionMultiplication : public QuaternionBinaryExpr<QL, QR>
	{
		/// The quaternion expression multiplier on the left-hand side
		QL l_;
		/// The quaternion expression multiplicand on the right-hand side
		QR r_;

	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion multiplication constructor taking 
		 * indepndent expressions for both multiplier and multiplicand.
		 * 
		 * @param lhs The quaternion expression on the left-hand side
		 * @param rhs The quaternion expression on the right-hand side
		 */ 
		QuaternionMultiplication(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return l_.w() * r_.w() - l_.x() * r_.x() - l_.y() * r_.y() - l_.z() * r_.z(); }
		Scalar x() const noexcept { return l_.w() * r_.x() + l_.x() * r_.w() + l_.y() * r_.z() - l_.z() * r_.y(); }
		Scalar y() const noexcept { return l_.w() * r_.y() - l_.x() * r_.z() + l_.y() * r_.w() + l_.z() * r_.x(); }
		Scalar z() const noexcept { return l_.w() * r_.z() + l_.x() * r_.y() - l_.y() * r_.x() + l_.z() * r_.w(); }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Multiplication Operator:
	 *       \f$\mathbf{q_1}*\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * multiplier, the factor on the left-hand side of the *.
	 * 
	 * @tparam QR The class of the quaternion expression that is the 
	 * multiplicand, the factor on the right-hand side of the *.
	 * 
	 * @param lhs The quternion expression multiplier on the left-hand 
	 * side of the *.
	 * 
	 * @param rhs The quaternion expression multiplicand on the 
	 * right-hand side of the *.
	 * 
	 * @return An instance of a QuaternionMultiplication template 
	 * expression tree node.
	 * 
	 * @details The binary `operator*` creates a template expression tree 
	 * node for evaluating the multiplication together of two 
	 * quaternions.
	 * 
	 * @include{doc} Math/Quaternion/Multiplication.txt
	 * 
	 * @sa ark::math::QuaternionMultiplication
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto operator*(const QL& lhs, const QR& rhs) noexcept -> QuaternionMultiplication<QL, QR>
	{
		return QuaternionMultiplication(lhs, rhs);
	}


	/*********************************************************************
	 * @brief Quaternion Dot Product Function:
	 *       \f$\mathbf{q_1}\cdot\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * first factor of the dot product.
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * second factor of the dot product.
	 * 
	 * @param lhs The quternion expression that is the first factor.
	 * 
	 * @param rhs The quaternion expression that is the second factor.
	 * 
	 * @return The value of the dot product.
	 * 
	 * @details @include{doc} Math/Quaternion/DotProduct.txt
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto Dot(const QL& lhs, const QR& rhs) noexcept -> typename QuaternionBinaryExpr<QL, QR>::Scalar
	{
		return
			lhs.w() * rhs.w() +
			lhs.x() * rhs.x() +
			lhs.y() * rhs.y() +
			lhs.z() * rhs.z();
	}


	/*********************************************************************
	 * @brief Quaternion Norm Function:
	 *       \f$\mathrm{Norm}(\mathbf{q})\f$
	 * 
	 * @tparam Q The class of the quaternion expression whose norm is to 
	 * be taken.
	 * 
	 * @param q The quaternion whose norm is to be computed.
	 * 
	 * @return The norm of the quaternion passed in as an argument.
	 * 
	 * @details Compute the norm of a given quaternion. This is the 
	 * Euclidean distance of the quaternion from the origin.
	 * 
	 * @include{doc} Math/Quaternion/Norm.f
	 * Given the 
	 * quaternion \f$\mathbf{q}\f$, its norm is computed like this:
	 * \f[\mathrm{Norm}(\mathbf{q})=\sqrt{\mathbf{q}\cdot\mathbf{q}}\f]
	 * 
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto Norm(const Q& q) noexcept -> typename QuaternionUnaryExpr<Q>::Scalar
	{
		return std::sqrt(Dot(q, q));
	}


	 /*********************************************************************
	 * @brief Quaternion Inversion Expression Node:
	 * \f$\mathbf{q}^{-1}\f$
	 * 
	 * @tparam Q The class of the quaternion expression whose value whose 
	 * multiplicative inverse is to be calculated.
	 * 
	 * @param q The quaternion value whose inverse is to be calculated.
	 * 
	 * @details @include{doc} Math/Quaternion/Inversion.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	class QuaternionInversion : public QuaternionUnaryExpr<Q>
	{
	public:
		using Scalar = typename Q::Scalar;

	private:
		/// The quaternion expression whose inverse is to be computed			
		Q q_;
		/// Cached result computed in the constructor
		QuaternionExpr::Cache<Scalar> cache_;

	public:
		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion inverse expression constructor. This is one 
		 * of the rare constructors that computes the value and stores it 
		 * in a cache for simple access later. The computation is much 
		 * too complex to be efficiently computed at each component 
		 * access.
		 * 
		 * @param q The quaternion whose inverse gets computed
		 */ 
		QuaternionInversion(const Q& q) noexcept
			: q_(q)
		{
			auto result = *q / Dot(q, q);
			cache_.w = result.w();
			cache_.x = result.x();
			cache_.y = result.y();
			cache_.z = result.z();
		}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return cache_.w; }
		Scalar x() const noexcept { return cache_.x; }
		Scalar y() const noexcept { return cache_.y; }
		Scalar z() const noexcept { return cache_.z; }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Inverse Function:
	 *       \f$\mathbf{q}^{-1}\f$
	 * 
	 * @tparam Q The class of the quaternion expression parameter.
	 *  
	 * @param q The quaternion whose inverse is to be taken.
	 * 
	 * @return The multiplicative inverse of the quaternion passed in.
	 * 
	 * @details Create a QuaternionInversion expression tree node to 
	 * lazily evalue the multiplicative inverse of a given quaternion.
	 *
	 * @include{doc} Math\Quaternion\Inversion.txt 
	 * 
	 * @sa ark::math::QuaternionInversion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion Q>
	inline auto Inverse(const Q& q) noexcept -> QuaternionInversion<Q>
	{
		return QuaternionInversion<Q>(q);
	}


	 /*********************************************************************
	 * @brief Quaternion Division Expression Node:
	 * \f$\mathbf{q_1}/\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * dividend of the division expression
	 * 
	 * @tparam QR The class of the quaternion expression that is the 
	 * divisor of the division expression.
	 * 
	 * @param lhs The quaternion value of the dividend
	 * @param rhs The quaternion value of hte divisor
	 * 
	 * @details This node computes quotient of one quaternion expression 
	 * divided by another. Due to the complexity of the computation it is 
	 * one of the few nodes that computes its value immediately and has 
	 * accessors that merely returned the cached values.
	 * 
	 * @include{doc} Math/Quaternion/Division.txt
	 * 
	 * @sa ark::math::Quaternion
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	class QuaternionDivision : public QuaternionBinaryExpr<QL, QR>
	{
	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

	private:
		/// Quaternione expression dividend
		QL l_;
		/// Quaternion expression divisor
		QR r_;
		/// Cached result of the division computation
		QuaternionExpr::Cache<Scalar> cache_;

	public:
		/// @name Constructors
		/// @{
		/**
		 * @brief Quaternion division expression constructor. This is one 
		 * of the rare constructors that computes the value and stores it 
		 * in a cache for simple access later. The computation is much 
		 * too complex to be efficiently computed at each component 
		 * access.
		 * 
		 * @param lhs The quaternion expression dividend value
		 * @param rhs The quaternion expression divisor value
		 */ 
		QuaternionDivision(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{
			auto result = lhs * Inverse(rhs);
			cache_.w = result.w();
			cache_.x = result.x();
			cache_.y = result.y();
			cache_.z = result.z();
		}
		/// @}

		/// @name Accessors
		/// @{
		Scalar w() const noexcept { return cache_.w; }
		Scalar x() const noexcept { return cache_.x; }
		Scalar y() const noexcept { return cache_.y; }
		Scalar z() const noexcept { return cache_.z; }
		/// @}
	};


	/*********************************************************************
	 * @brief Quaternion Division Operator:
	 *       \f$\mathbf{q_1}/\mathbf{q_2}\f$
	 * 
	 * @tparam QL The class of the quaternion expression that is the 
	 * dividend, the factor on the left-hand side of the /.
	 * 
	 * @tparam QR The class of the quaternion expression that is the 
	 * divisor, the factor on the right-hand side of the /.
	 * 
	 * @param lhs The quternion expression dividend on the left-hand 
	 * side of the /.
	 * 
	 * @param rhs The quaternion expression divisor on theright-hand side 
	 * of the /.
	 * 
	 * @return An instance of a QuaternionDivision template 
	 * expression tree node to evaluate the division expression.
	 * 
	 * @details The binary `operator/`creates a template expression tree 
	 * node for evaluating the division of the dividend by the divisor 
	 * quaternions. Unlike most quaternion expression nodes, this one 
	 * evaluates the result early and caches the data for efficiency
	 * purposes; otherwise, each accessor would be required to compute 
	 * the value separately, which many multipliy and add steps, a 
	 * significant amount of redundant computation as it is most likely 
	 * all four accessor will be called at least once.
	 * 
	 * @include{doc} Math/Quaternion/Division.txt
	 * 
	 * @sa ark::math::QuaternionDivision
	 * @sa @ref ExpressionTemplates
	 ********************************************************************/
	template<Quaternion QL, Quaternion QR>
	inline auto operator/(const QL& lhs, const QR& rhs) noexcept -> QuaternionDivision<QL, QR>
	{
		return QuaternionDivision(lhs, rhs);
	}
}


//************************************************************************
#endif
