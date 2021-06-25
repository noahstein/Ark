/*========================================================================
Description
	Defines the Quaternion concept to represent the abstract definition
	of the structure of a mathematical quaternion. From there, this file 
	implements many common quaternion functions on that abstraction. Due 
	to this implementation to the abstraction, each new concrete 
	quaternion class oinly needs to define the type of its scalars and 
	accessors to its elements, and it will not only automatically have 
	a full set of operations, it will interact with any other mix of 
	quaternion classes, as long as the scalar types may interact 
	arithmetically.
	
	Additionally, any specific quaternion class may define its own 
	operators and implement optimized code for those oeprations. Classes 
	may implement any operators they so choose and still rely upon the 
	abstract implementations for the rest.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/

#if !defined(ARK_MATH_QUATERNION_H_INCLUDE_GUARD)
#define ARK_MATH_QUATERNION_H_INCLUDE_GUARD


/*========================================================================
	Dependencies
========================================================================*/
#include <cmath>
#include <type_traits>

#include "Concepts.h"


//========================================================================
//	Code
//========================================================================
namespace ark::math
{
	/*====================================================================
		Concepts
	====================================================================*/

	/*--------------------------------------------------------------------
		Quaternion Concept
	--------------------------------------------------------------------*/
	template<typename Q>
	concept Quaternion = requires(Q q)
	{
		typename Q::Scalar;
		Arithmetic<typename Q::Scalar>;

		// Accessor to the w, x, y, and z components
		{ q.w() } -> std::same_as<typename Q::Scalar>;
		{ q.x() } -> std::same_as<typename Q::Scalar>;
		{ q.y() } -> std::same_as<typename Q::Scalar>;
		{ q.z() } -> std::same_as<typename Q::Scalar>;
	};


	/*====================================================================
		Expression Base Classes
	====================================================================*/

	/*--------------------------------------------------------------------
		Base class for all quaternion expressions.
	--------------------------------------------------------------------*/
	class QuaternionExpr
	{
	protected:
		template<typename S>
		struct Cache
		{
			S w, x, y, z;
		};
	};


	/*--------------------------------------------------------------------
		The base class of all quaternion 1-argument expressions.
	//------------------------------------------------------------------*/
	template<Quaternion Q>
	class QuaternionUnaryExpr : public QuaternionExpr
	{
	public:
		using Scalar = typename Q::Scalar;
	};


	//----------------------------------------------------------------
	//	The base class for all quaternion 2-argument expressions
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
		requires MutuallyArithmetic<typename QL::Scalar, typename QR::Scalar>
	class QuaternionBinaryExpr : public QuaternionExpr
	{
		using SL = typename QL::Scalar;
		using SR = typename QR::Scalar;

	public:
		using Scalar = typename std::common_type<SL, SR>::type;
	};


	/*====================================================================
		Expression Function Classes
	====================================================================*/

	/*--------------------------------------------------------------------
		Quaternion Negation Expression: -q
	//------------------------------------------------------------------*/
	template<Quaternion Q>
	class QuaternionNegation : QuaternionUnaryExpr<Q>
	{
		Q q_;

	public:
		using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

		QuaternionNegation(const Q& q) noexcept
			: q_(q)
		{}

		Scalar w() const noexcept { return -q_.w(); }
		Scalar x() const noexcept { return -q_.x(); }
		Scalar y() const noexcept { return -q_.y(); }
		Scalar z() const noexcept { return -q_.z(); }
	};


	/*--------------------------------------------------------------------
		Quaternion Negation Operator
	--------------------------------------------------------------------*/
	template<Quaternion Q>
	inline auto operator-(const Q& q) noexcept -> QuaternionNegation<Q>
	{
		return QuaternionNegation(q);
	}


	//----------------------------------------------------------------
	//	Quaternion Conjugate Expression: *q (q* in the books)
	//----------------------------------------------------------------
	template<typename Q>
	class QuaternionConjugate : QuaternionUnaryExpr<Q>
	{
		Q q_;

	public:
		using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

		QuaternionConjugate(const Q& q) noexcept
			: q_(q)
		{}

		Scalar w() const noexcept { return  q_.w(); }
		Scalar x() const noexcept { return -q_.x(); }
		Scalar y() const noexcept { return -q_.y(); }
		Scalar z() const noexcept { return -q_.z(); }
	};


	//----------------------------------------------------------------
	//	Quaternion Conjugate Operator
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto operator*(const Q& q) noexcept -> QuaternionConjugate<Q>
	{
		return QuaternionConjugate(q);
	}


	//----------------------------------------------------------------
	//	Quaternion Addition Expression: q1 + q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	class QuaternionAddition : public QuaternionBinaryExpr<QL, QR>
	{
		QL l_;
		QR r_;

	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		QuaternionAddition(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		Scalar w() const noexcept { return l_.w() + r_.w(); }
		Scalar x() const noexcept { return l_.x() + r_.x(); }
		Scalar y() const noexcept { return l_.y() + r_.y(); }
		Scalar z() const noexcept { return l_.z() + r_.z(); }
	};


	//----------------------------------------------------------------
	//	Quaternion Addition Operator
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto operator+(const QL& lhs, const QR& rhs) noexcept -> QuaternionAddition<QL, QR>
	{
		return QuaternionAddition(lhs, rhs);
	}


	//----------------------------------------------------------------
	//	Quaternion Subtraction Expression: q1 - q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	class QuaternionSubtraction : public QuaternionBinaryExpr<QL, QR>
	{
		QL l_;
		QR r_;

	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		QuaternionSubtraction(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		Scalar w() const noexcept { return l_.w() - r_.w(); }
		Scalar x() const noexcept { return l_.x() - r_.x(); }
		Scalar y() const noexcept { return l_.y() - r_.y(); }
		Scalar z() const noexcept { return l_.z() - r_.z(); }
	};


	//----------------------------------------------------------------
	//	Quaternion Subtraction Operator
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto operator-(const QL& lhs, const QR& rhs) noexcept -> QuaternionSubtraction<QL, QR>
	{
		return QuaternionSubtraction(lhs, rhs);
	}


	//----------------------------------------------------------------
	//	Quaternion Scalar Multiplication Expression: s * q, q *s
	//----------------------------------------------------------------
	template<Quaternion Q>
	class QuaternionScalarMultiplication : public QuaternionExpr
	{
	public:
		using Scalar = typename Q::Scalar;

	private:
		Scalar s_;
		Q q_;

	public:
		QuaternionScalarMultiplication(Scalar s, const Q& q) noexcept
			: s_(s), q_(q)
		{}

		Scalar w() const noexcept { return s_ * q_.w(); }
		Scalar x() const noexcept { return s_ * q_.x(); }
		Scalar y() const noexcept { return s_ * q_.y(); }
		Scalar z() const noexcept { return s_ * q_.z(); }
	};


	//----------------------------------------------------------------
	//	Quaternion Scalar-Quaternion Multiplication Operator
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto operator*(typename Q::Scalar s, const Q& q) noexcept -> QuaternionScalarMultiplication<Q>
	{
		return QuaternionScalarMultiplication(s, q);
	}


	//----------------------------------------------------------------
	//	Quaternion Quaternion-Scalar Multiplication Operator
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto operator*(const Q& q, typename Q::Scalar s) noexcept -> QuaternionScalarMultiplication<Q>
	{
		return QuaternionScalarMultiplication(s, q);
	}


	//----------------------------------------------------------------
	//	Quaternion Scalar Division Expression: q / s
	//----------------------------------------------------------------
	template<Quaternion Q>
	class QuaternionScalarDivision : public QuaternionUnaryExpr<Q>
	{
	public:
		using Scalar = typename Q::Scalar;

	private:
		Q q_;
		Scalar s_;

	public:
		QuaternionScalarDivision(const Q& q, Scalar s) noexcept
			: q_(q), s_(s)
		{}

		Scalar w() const noexcept { return q_.w() / s_; }
		Scalar x() const noexcept { return q_.x() / s_; }
		Scalar y() const noexcept { return q_.y() / s_; }
		Scalar z() const noexcept { return q_.z() / s_; }
	};


	//----------------------------------------------------------------
	//	Quaternion Scalar Division Operator
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto operator/(const Q& q, typename Q::Scalar s) noexcept -> QuaternionScalarDivision<Q>
	{

		return QuaternionScalarDivision(q, s);
	}
	
	//----------------------------------------------------------------
	//	Quaternion Equality Operator: q1 == q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto operator==(const QL& lhs, const QR& rhs) noexcept -> bool
	{
		return
			lhs.w() == rhs.w() && 
			lhs.x() == rhs.x() &&
			lhs.y() == rhs.y() &&
			lhs.z() == rhs.z();
	}


	//----------------------------------------------------------------
	//	Quaternion Multiplication Expression: q1 * q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	class QuaternionMultiplication : public QuaternionBinaryExpr<QL, QR>
	{
		QL l_;
		QR r_;

	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

		QuaternionMultiplication(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{}

		Scalar w() const noexcept { return l_.w() * r_.w() - l_.x() * r_.x() - l_.y() * r_.y() - l_.z() * r_.z(); }
		Scalar x() const noexcept { return l_.w() * r_.x() + l_.x() * r_.w() + l_.y() * r_.z() - l_.z() * r_.y(); }
		Scalar y() const noexcept { return l_.w() * r_.y() - l_.x() * r_.z() + l_.y() * r_.w() + l_.z() * r_.x(); }
		Scalar z() const noexcept { return l_.w() * r_.z() + l_.x() * r_.y() - l_.y() * r_.x() + l_.z() * r_.w(); }
	};


	//----------------------------------------------------------------
	//	Quaternion Multiplication Operator
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto operator*(const QL& lhs, const QR& rhs) noexcept -> QuaternionMultiplication<QL, QR>
	{
		return QuaternionMultiplication(lhs, rhs);
	}


	//----------------------------------------------------------------
	//	Dot Product Function: Dot(q)
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto Dot(const QL& l, const QR& r) noexcept -> typename QuaternionBinaryExpr<QL, QR>::Scalar
	{
		return
			l.w() * r.w() +
			l.x() * r.x() +
			l.y() * r.y() +
			l.z() * r.z();
	}


	//----------------------------------------------------------------
	//	Norm Function: Norm(q)
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto Norm(const Q& q) noexcept -> typename QuaternionUnaryExpr<Q>::Scalar
	{
		return std::sqrt(Dot(q, q));
	}


	//----------------------------------------------------------------
	//	Quaternion Inverse Expression: q^-1
	//----------------------------------------------------------------
	template<Quaternion Q>
	class QuaternionInversion : public QuaternionUnaryExpr<Q>
	{
	public:
		using Scalar = typename Q::Scalar;

	private:			
		Q q_;
		QuaternionExpr::Cache<Scalar> cache_;	// TODO: "QuaternionExpr" should be unneceessary.

	public:
		QuaternionInversion(const Q& q) noexcept
			: q_(q)
		{
			auto result = *q / Dot(q, q);
			cache_.w = result.w();
			cache_.x = result.x();
			cache_.y = result.y();
			cache_.z = result.z();
		}

		Scalar w() const noexcept { return cache_.w; }
		Scalar x() const noexcept { return cache_.x; }
		Scalar y() const noexcept { return cache_.y; }
		Scalar z() const noexcept { return cache_.z; }
	};


	//----------------------------------------------------------------
	//	Quaternion Inverse Function: Inverse(q)
	//----------------------------------------------------------------
	template<Quaternion Q>
	inline auto Inverse(const Q& q) noexcept -> QuaternionInversion<Q>
	{
		return QuaternionInversion<Q>(q);
	}


	//----------------------------------------------------------------
	//	Quaternion Division Expression: q1 / q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	class QuaternionDivision : public QuaternionBinaryExpr<QL, QR>
	{
	public:
		using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

	private:			
		QL l_;
		QR r_;
		QuaternionExpr::Cache<Scalar> cache_;	// TODO: "QuaternionExpr" should be unneceessary.

	public:
		QuaternionDivision(const QL& lhs, const QR& rhs) noexcept
			: l_(lhs), r_(rhs)
		{
			auto result = lhs * Inverse(rhs);
			cache_.w = result.w();
			cache_.x = result.x();
			cache_.y = result.y();
			cache_.z = result.z();
		}

		Scalar w() const noexcept { return cache_.w; }
		Scalar x() const noexcept { return cache_.x; }
		Scalar y() const noexcept { return cache_.y; }
		Scalar z() const noexcept { return cache_.z; }
	};


	//----------------------------------------------------------------
	//	Quaternion Division Operator: q1 / q2
	//----------------------------------------------------------------
	template<Quaternion QL, Quaternion QR>
	inline auto operator/(const QL& lhs, const QR& rhs) noexcept -> QuaternionDivision<QL, QR>
	{
		return QuaternionDivision(lhs, rhs);
	}
}


//========================================================================
#endif
