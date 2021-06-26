/*========================================================================
Description
	Unit tests for functions related to the ark::math::Quaternion concept, 
	and all basic functionality implemented against that concept.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/


/*========================================================================
 Dependencies
========================================================================*/
#include "gtest/gtest.h"
#include "Ark/Math/Quaternion.h"


/*========================================================================
 Code
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
		Test Quaternion class implements the Quaternion concept in a 
		simple way in order to test the functionality implemented against 
		the concept
	--------------------------------------------------------------------*/
	template<typename S>
	class TestQuat
	{
	public:
		using Scalar = S;

	private:
		Scalar w_, x_, y_, z_;

	public:
		TestQuat()
		{}

		TestQuat(Scalar ww, Scalar xx, Scalar yy, Scalar zz)
			: w_(ww), x_(xx), y_(yy), z_(zz)
		{}

		template<ark::math::Quaternion Q>
		TestQuat(const Q& rhs)
			: TestQuat(Scalar{rhs.w()}, Scalar{rhs.x()}, Scalar{rhs.y()}, Scalar{rhs.z()})
		{}

		template<ark::math::Quaternion Q>
			requires std::is_convertible_v<typename Q::Scalar, Scalar>
		TestQuat& operator=(const Q& rhs)
		{
			x_ = Scalar(rhs.x());
			y_ = Scalar(rhs.y());
			w_ = Scalar(rhs.w());
			z_ = Scalar(rhs.z());

			return *this;
		}

		Scalar w() const { return w_; }
		Scalar x() const { return x_; }
		Scalar y() const { return y_; }
		Scalar z() const { return z_; }
	};
}


namespace ark::math::test
{
	/*--------------------------------------------------------------------
	 Quaternion Test Fixture
	--------------------------------------------------------------------*/
	class QuaternionUnitTests : public testing::Test
	{
	protected:
		void SetUp() override
		{

			q_one = TestQuat(1, 0, 0, 0);
			qi = TestQuat(0, 1, 0, 0);
			qj = TestQuat(0, 0, 1, 0);
			qk = TestQuat(0, 0, 0, 1);

			q1 = TestQuat(3, 13, 7, 19);
			q2 = TestQuat(5, 11, 23, 29);
		}

		TestQuat<float> q1;
		TestQuat<float> q2;
		TestQuat<float> qr;

		TestQuat<float> q_one;
		TestQuat<float> qi;
		TestQuat<float> qj;
		TestQuat<float> qk;
	};



	//========================================================================
	// Tests
	//========================================================================

	TEST_F(QuaternionUnitTests, Negate)
	{
		// When
		qr = -q1;

		// Then
		EXPECT_EQ(qr.w(), -3);
		EXPECT_EQ(qr.x(), -13);
		EXPECT_EQ(qr.y(), -7);
		EXPECT_EQ(qr.z(), -19);
	}


	TEST_F(QuaternionUnitTests, Conjugate)
	{
		// When
		qr = *q1;

		// Then
		EXPECT_EQ(qr.w(), 3);
		EXPECT_EQ(qr.x(), -13);
		EXPECT_EQ(qr.y(), -7);
		EXPECT_EQ(qr.z(), -19);
	}

#if 0
	TEST_F(QuaternionUnitTests, Addition)
	{
		// When
		qr = q1 + q2;

		// Then
		EXPECT_EQ(qr.w(), 8);
		EXPECT_EQ(qr.x(), 24);
		EXPECT_EQ(qr.y(), 30);
		EXPECT_EQ(qr.z(), 48);
	}


	TEST_F(QuaternionUnitTests, Subtraction)
	{
		// When
		qr = q2 - q1;

		// Then
		EXPECT_EQ(qr.w(), 2);
		EXPECT_EQ(qr.x(), -2);
		EXPECT_EQ(qr.y(), 16);
		EXPECT_EQ(qr.z(), 10);
	}


	TEST_F(QuaternionUnitTests, ScalarQuaternionMultiplication)
	{
		// When
		qr = 5 * q1;

		// Then
		EXPECT_EQ(qr.w(), 15);
		EXPECT_EQ(qr.x(), 65);
		EXPECT_EQ(qr.y(), 35);
		EXPECT_EQ(qr.z(), 95);
	}


	TEST_F(QuaternionUnitTests, QuaternionScalarMultiplication)
	{
		// When
		qr = q1 * 3;

		// Then
		EXPECT_EQ(qr.w(), 9);
		EXPECT_EQ(qr.x(), 39);
		EXPECT_EQ(qr.y(), 21);
		EXPECT_EQ(qr.z(), 57);
	}


	TEST_F(QuaternionUnitTests, QuaternionScalarDivision)
	{
		// When
		qr = q1 / 2;

		// Then
		EXPECT_EQ(qr.w(), 1.5);
		EXPECT_EQ(qr.x(), 6.5);
		EXPECT_EQ(qr.y(), 3.5);
		EXPECT_EQ(qr.z(), 9.5);
	}


	TEST_F(QuaternionUnitTests, InequalityCheckSame)
	{
		// When
		bool result = q1 != q1;

		// Then
		EXPECT_FALSE(result);
	}


	TEST_F(QuaternionUnitTests, InequalityCheckDifferent)
	{
		// When
		bool result = q1 != q2;

		// Then
		EXPECT_TRUE(result);
	}


	TEST_F(QuaternionUnitTests, I_x_I_eq_MinusOne)
	{
		// When
		qr = qi * qi;

		// Then
		EXPECT_EQ(qr, -q_one);
	}


	TEST_F(QuaternionUnitTests, J_x_J_eq_MinusOne)
	{
		// When
		qr = qj * qj;

		// Then
		EXPECT_EQ(qr, -q_one);
	}


	TEST_F(QuaternionUnitTests, K_x_K_eq_MinusOne)
	{
		// When
		qr = qk * qk;

		// Then
		EXPECT_EQ(qr, -q_one);
	}


	TEST_F(QuaternionUnitTests, I_x_J_eq_K)
	{
		// When
		qr = qi * qj;

		// Then
		EXPECT_EQ(qr, qk);
	}


	TEST_F(QuaternionUnitTests, J_x_K_eq_I)
	{
		// When
		qr = qj * qk;

		// Then
		EXPECT_EQ(qr, qi);
	}


	TEST_F(QuaternionUnitTests, K_x_I_eq_J)
	{
		// When
		qr = qk * qi;

		// Then
		EXPECT_EQ(qr, qj);
	}


	TEST_F(QuaternionUnitTests, J_x_I_eq_MinusK)
	{
		// When
		qr = qj * qi;

		// Then
		EXPECT_EQ(qr, -qk);
	}


	TEST_F(QuaternionUnitTests, K_x_J_eq_MinusI)
	{
		// When
		qr = qk * qj;

		// Then
		EXPECT_EQ(qr, -qi);
	}


	TEST_F(QuaternionUnitTests, I_x_K_eq_MinusJ)
	{
		// When
		qr = qi * qk;

		// Then
		EXPECT_EQ(qr, -qj);
	}


	TEST_F(QuaternionUnitTests, DotProduct)
	{
		// When
		float r = Dot(q1, q2);

		// Then
		EXPECT_EQ(r, 870);
	}


	TEST_F(QuaternionUnitTests, Norm)
	{
		// When
		float r = Norm(q1);

		// Then
		EXPECT_FLOAT_EQ(r, 24.24871130596428f);
	}


	TEST_F(QuaternionUnitTests, LeftInverse)
	{
		// When
		qr = Inverse(q1) * q1;

		// Then
		EXPECT_NEAR(qr.w(), 1, 0.00001);
		EXPECT_NEAR(qr.x(), 0, 0.00001);
		EXPECT_NEAR(qr.y(), 0, 0.00001);
		EXPECT_NEAR(qr.z(), 0, 0.00001);
	}


	TEST_F(QuaternionUnitTests, RightInverse)
	{
		// When
		qr = q1 * Inverse(q1);

		// Then
		EXPECT_NEAR(qr.w(), 1, 0.00001);
		EXPECT_NEAR(qr.x(), 0, 0.00001);
		EXPECT_NEAR(qr.y(), 0, 0.00001);
		EXPECT_NEAR(qr.z(), 0, 0.00001);
	}


	TEST_F(QuaternionUnitTests, Division)
	{
		// When
		qr = q1 / q1;

		// Then
		EXPECT_NEAR(qr.w(), 1, 0.00001);
		EXPECT_NEAR(qr.x(), 0, 0.00001);
		EXPECT_NEAR(qr.y(), 0, 0.00001);
		EXPECT_NEAR(qr.z(), 0, 0.00001);
	}
#endif
}
