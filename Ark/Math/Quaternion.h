
#pragma once

#include <concepts>
#include <type_traits>

namespace ark
{
	namespace math
	{
		template<typename Q>
		concept Quaternion = requires(Q q)
		{
			typename Q::Scalar;	// The type of the individual X, Y, Z, and W components.

			{ q.w() } -> std::same_as<typename Q::Scalar>;
			{ q.x() } -> std::same_as<typename Q::Scalar>;
			{ q.y() } -> std::same_as<typename Q::Scalar>;
			{ q.z() } -> std::same_as<typename Q::Scalar>;
		};

		class QuaternionExpr
		{
		};

	
		template<class Q>
		class QuaternionUnaryExpr : public QuaternionExpr
		{
		public:
			using Scalar = typename Q::Scalar;
		};

		template<typename Q>
		class QuaternionNegation : QuaternionUnaryExpr<Q>
		{
			Q q_;

		public:
			using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

			QuaternionNegation(const Q& qq)
				: q_(qq)
			{}

			Scalar w() const { return -q_.w(); }
			Scalar x() const { return -q_.x(); }
			Scalar y() const { return -q_.y(); }
			Scalar z() const { return -q_.z(); }
		};

		template<Quaternion Q>
		auto operator-(const Q& q) -> QuaternionNegation<Q>
		{
			return QuaternionNegation(q);
		}

		template<typename Q>
		class QuaternionConjugate : QuaternionUnaryExpr<Q>
		{
			Q q_;

		public:
			using Scalar = typename QuaternionUnaryExpr<Q>::Scalar;

			QuaternionConjugate(const Q& qq)
				: q_(qq)
			{}

			Scalar w() const { return q_.w(); }
			Scalar x() const { return -q_.x(); }
			Scalar y() const { return -q_.y(); }
			Scalar z() const { return -q_.z(); }
		};

		template<Quaternion Q>
		auto operator*(const Q& q) -> QuaternionConjugate<Q>
		{
			return QuaternionConjugate(q);
		}

		template<typename QL, typename QR>
		class QuaternionBinaryExpr : public QuaternionExpr
		{
			using SL = typename QL::Scalar;
			using SR = typename QR::Scalar;

		public:
			using Scalar = typename std::common_type<SL, SR>::type;
		};

		template<Quaternion QL, Quaternion QR>
		class QuaternionAddition : public QuaternionBinaryExpr<QL, QR>
		{
			QL l_;
			QR r_;

		public:
			using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

			QuaternionAddition(const QL& lhs, const QR& rhs)
				: l_(lhs), r_(rhs)
			{}

			Scalar w() const { return l_.w() + r_.w(); }
			Scalar x() const { return l_.x() + r_.x(); }
			Scalar y() const { return l_.y() + r_.y(); }
			Scalar z() const { return l_.z() + r_.z(); }
		};

		template<Quaternion QL, Quaternion QR>
		auto operator+(const QL& lhs, const QR& rhs) -> QuaternionAddition<QL, QR>
		{
			return QuaternionAddition(lhs, rhs);
		}

		template<Quaternion QL, Quaternion QR>
		class QuaternionSubtraction : public QuaternionBinaryExpr<QL, QR>
		{
			QL l_;
			QR r_;

		public:
			using Scalar = typename QuaternionBinaryExpr<QL, QR>::Scalar;

			QuaternionSubtraction(const QL& lhs, const QR& rhs)
				: l_(lhs), r_(rhs)
			{}

			Scalar w() const { return l_.w() - r_.w(); }
			Scalar x() const { return l_.x() - r_.x(); }
			Scalar y() const { return l_.y() - r_.y(); }
			Scalar z() const { return l_.z() - r_.z(); }
		};

		template<Quaternion QL, Quaternion QR>
		auto operator-(const QL& lhs, const QR& rhs) -> QuaternionSubtraction<QL, QR>
		{
			return QuaternionSubtraction(lhs, rhs);
		}

		template<typename S>
		class Quat
		{
			S w_, x_, y_, z_;

		public:
			using Scalar = S;

			Quat()
			{}

			Quat(Scalar ww, Scalar xx, Scalar yy, Scalar zz)
				: w_(ww), x_(xx), y_(yy), z_(zz)
			{}

			template<Quaternion Q>
			Quat(const Q& rhs)
				: Quat(static_cast<S>(rhs.w()), static_cast<S>(rhs.x()), static_cast<S>(rhs.y()), static_cast<S>(rhs.z()))
			{}

			template<Quaternion Q>
			// Put in convertible test from Q::S -> S
			Quat& operator=(const Q& rhs)
			{
				w_ = Scalar(rhs.w());
				x_ = Scalar(rhs.x());
				y_ = Scalar(rhs.y());
				z_ = Scalar(rhs.z());

				return *this;
			}

			Scalar w() const { return w_; }
			Scalar x() const { return x_; }
			Scalar y() const { return y_; }
			Scalar z() const { return z_; }
		};
	}
}

#include "SSE/Quaternion_SSE.h"
