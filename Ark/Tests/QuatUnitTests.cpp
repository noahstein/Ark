/*========================================================================
Description
	Unit tests for the Quat class implementing the Quaternion concept.
	As most functions are implemented against the concept, they need not
	be tested hwere. What's important is testing Quat's actual member
	functions.

	The other important aspect is to test any functionality that is
	overridden in specializations of the class. For example, Quat<float>
	has platform-specific specializations to use SIMD hardware;
	therefore, there are tests here to ensure those work properly.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/


//========================================================================
// Dependencies
//========================================================================
#include "gtest/gtest.h"
#include "Ark/Math/Quat.h"

template<typename S> using Quat = ark::math::Quat<S, ark::hal::simd::HAL_SIMD>;


//========================================================================
// Fixtures
//========================================================================

//------------------------------------------------------------------------
// Quat<float> Test Fixture
//------------------------------------------------------------------------
class QuatFloatUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		q1 = Quat<float>(3, 13, 7, 19);
		q2 = Quat<float>(5, 11, 23, 29);
	}

	const Quat<float> q_one{ 1.0f, 0.0f, 0.0f, 0.0f };
	const Quat<float> qi{ 0.0f, 1.0f, 0.0f, 0.0f };
	const Quat<float> qj{ 0.0f, 0.0f, 1.0f, 0.0f };
	const Quat<float> qk{ 0.0f, 0.0f, 0.0f, 1.0f };

	Quat<float> q1;
	Quat<float> q2;
	Quat<float> qr;
};


//========================================================================
// Tests
//========================================================================

TEST(QuatUnitTests, DefaultConstructor)
{
	Quat<int> q;
	SUCCEED();
}


TEST(QuatUnitTests, ElementConstructor)
{
	Quat<int> q(3, 5, 7, 11);
	EXPECT_EQ(q.w(), 3) << "The w element";
	EXPECT_EQ(q.x(), 5) << "The x element";
	EXPECT_EQ(q.y(), 7) << "The y element";
	EXPECT_EQ(q.z(), 11) << "The z element";
}


//------------------------------------------------------------------------
// Quat<float> Tests
//------------------------------------------------------------------------

TEST_F(QuatFloatUnitTests, Negate)
{
	// When
	qr = -q1;

	// Then
	EXPECT_EQ(qr.w(), -3);
	EXPECT_EQ(qr.x(), -13);
	EXPECT_EQ(qr.y(), -7);
	EXPECT_EQ(qr.z(), -19);
}


TEST_F(QuatFloatUnitTests, Conjugate)
{
	// When
	qr = *q1;

	// Then
	EXPECT_EQ(qr.w(), 3);
	EXPECT_EQ(qr.x(), -13);
	EXPECT_EQ(qr.y(), -7);
	EXPECT_EQ(qr.z(), -19);
}


TEST_F(QuatFloatUnitTests, LeftInverse)
{
	// When
	qr = Inverse(q1) * q1;

	// Then
	EXPECT_NEAR(qr.w(), 1, 0.00001);
	EXPECT_NEAR(qr.x(), 0, 0.00001);
	EXPECT_NEAR(qr.y(), 0, 0.00001);
	EXPECT_NEAR(qr.z(), 0, 0.00001);
}


TEST_F(QuatFloatUnitTests, RightInverse)
{
	// When
	qr = q1 * Inverse(q1);

	// Then
	EXPECT_NEAR(qr.w(), 1, 0.00001);
	EXPECT_NEAR(qr.x(), 0, 0.00001);
	EXPECT_NEAR(qr.y(), 0, 0.00001);
	EXPECT_NEAR(qr.z(), 0, 0.00001);
}


TEST_F(QuatFloatUnitTests, Addition)
{
	// When
	qr = q1 + q2;

	// Then
	EXPECT_EQ(qr.w(), 8);
	EXPECT_EQ(qr.x(), 24);
	EXPECT_EQ(qr.y(), 30);
	EXPECT_EQ(qr.z(), 48);
}


TEST_F(QuatFloatUnitTests, Subtraction)
{
	// When
	qr = q2 - q1;

	// Then
	EXPECT_EQ(qr.w(), 2);
	EXPECT_EQ(qr.x(), -2);
	EXPECT_EQ(qr.y(), 16);
	EXPECT_EQ(qr.z(), 10);
}


TEST_F(QuatFloatUnitTests, ScalarQuaternionMultiplication)
{
	// When
	qr = 5 * q1;

	// Then
	EXPECT_EQ(qr.w(), 15);
	EXPECT_EQ(qr.x(), 65);
	EXPECT_EQ(qr.y(), 35);
	EXPECT_EQ(qr.z(), 95);
}


TEST_F(QuatFloatUnitTests, QuaternionScalarMultiplication)
{
	// When
	qr = q1 * 3;

	// Then
	EXPECT_EQ(qr.w(), 9);
	EXPECT_EQ(qr.x(), 39);
	EXPECT_EQ(qr.y(), 21);
	EXPECT_EQ(qr.z(), 57);
}


TEST_F(QuatFloatUnitTests, QuaternionScalarDivision)
{
	// When
	qr = q1 / 2;

	// Then
	EXPECT_EQ(qr.w(), 1.5);
	EXPECT_EQ(qr.x(), 6.5);
	EXPECT_EQ(qr.y(), 3.5);
	EXPECT_EQ(qr.z(), 9.5);
}


TEST_F(QuatFloatUnitTests, I_x_I_eq_MinusOne)
{
	// When
	qr = qi * qi;

	// Then
	EXPECT_EQ(qr, -q_one);
}


TEST_F(QuatFloatUnitTests, J_x_J_eq_MinusOne)
{
	// When
	qr = qj * qj;

	// Then
	EXPECT_EQ(qr, -q_one);
}


TEST_F(QuatFloatUnitTests, K_x_K_eq_MinusOne)
{
	// When
	qr = qk * qk;

	// Then
	EXPECT_EQ(qr, -q_one);
}


TEST_F(QuatFloatUnitTests, I_x_J_eq_K)
{
	// When
	qr = qi * qj;

	// Then
	EXPECT_EQ(qr, qk);
}


TEST_F(QuatFloatUnitTests, J_x_K_eq_I)
{
	// When
	qr = qj * qk;

	// Then
	EXPECT_EQ(qr, qi);
}


TEST_F(QuatFloatUnitTests, K_x_I_eq_J)
{
	// When
	qr = qk * qi;

	// Then
	EXPECT_EQ(qr, qj);
}


TEST_F(QuatFloatUnitTests, J_x_I_eq_MinusK)
{
	// When
	qr = qj * qi;

	// Then
	EXPECT_EQ(qr, -qk);
}


TEST_F(QuatFloatUnitTests, K_x_J_eq_MinusI)
{
	// When
	qr = qk * qj;

	// Then
	EXPECT_EQ(qr, -qi);
}


TEST_F(QuatFloatUnitTests, I_x_K_eq_MinusJ)
{
	// When
	qr = qi * qk;

	// Then
	EXPECT_EQ(qr, -qj);
}


TEST_F(QuatFloatUnitTests, Division)
{
	// When
	qr = q1 / q1;

	// Then
	EXPECT_NEAR(qr.w(), 1, 0.00001);
	EXPECT_NEAR(qr.x(), 0, 0.00001);
	EXPECT_NEAR(qr.y(), 0, 0.00001);
	EXPECT_NEAR(qr.z(), 0, 0.00001);
}
