/*========================================================================
Description
	Unit tests for functions related to the ark::math::Quaternion concept, 
	and all basic functionality implemented against that concept.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
========================================================================*/


//========================================================================
// Dependencies
//========================================================================
#include "gtest/gtest.h"
#include "Ark/Math/Quaternion.h"


//========================================================================
// Local Class Definitions
//========================================================================
namespace ark
{
	namespace math
	{
		//----------------------------------------------------------------
		// Test Quaternion class implements the Quaternion concept in 
		// a simple way in order to test the functionality implemented 
		// agains the concept
		//----------------------------------------------------------------
		class TestQuat
		{
			int w_, x_, y_, z_;

		public:
			using Scalar = int;

			TestQuat()
			{}

			TestQuat(int ww, int xx, int yy, int zz)
				: w_(ww), x_(xx), y_(yy), z_(zz)
			{}

			template<ark::math::Quaternion Q>
			TestQuat(const Q& rhs)
				: TestQuat(static_cast<int>(rhs.w()), static_cast<int>(rhs.x()), static_cast<int>(rhs.y()), static_cast<int>(rhs.z()))
			{}

			template<ark::math::Quaternion Q>
			// Put in convertible test from Q::S -> S
			TestQuat& operator=(const Q& rhs)
			{
				x_ = int(rhs.x());
				y_ = int(rhs.y());
				w_ = int(rhs.w());
				z_ = int(rhs.z());

				return *this;
			}

			int w() const { return w_; }
			int x() const { return x_; }
			int y() const { return y_; }
			int z() const { return z_; }
		};
	}
}
using Quat = ark::math::TestQuat;

//------------------------------------------------------------------------
// Quaternion Test Fixture
//------------------------------------------------------------------------
class QuaternionUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		q1 = Quat(3, 13, 7, 19);
		q2 = Quat(5, 11, 23, 29);
	}

	Quat q1;
	Quat q2;
	Quat qr;
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
