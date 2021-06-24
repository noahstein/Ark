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


/*========================================================================
  Dependencies
========================================================================*/
#include "gtest/gtest.h"
#include "Ark/Math/Quat.h"

template<typename S> using Quat = ark::math::Quat<S, ark::hal::simd::HAL_SIMD>;


/*========================================================================
  Test Suite
========================================================================*/

/*------------------------------------------------------------------------
  Quat Parametric Test Fixture
------------------------------------------------------------------------*/
template<typename C>
class QuatUnitTest : public testing::Test
{
protected:
	using T = typename C::T;
	using S = typename C::S;

	template<typename U>
	T t(U u)
	{
		return static_cast<T>(u);
	}

	void SetUp() override
	{
		q1 = ark::math::Quat<T, S>(3, 13, 7, 19);
		q2 = ark::math::Quat<T, S>(5, 11, 23, 29);
	}

	const ark::math::Quat<T, S> q_one{ 1, 0, 0, 0 };
	const ark::math::Quat<T, S> qi{ 0, 1, 0, 0 };
	const ark::math::Quat<T, S> qj{ 0, 0, 1, 0 };
	const ark::math::Quat<T, S> qk{ 0, 0, 0, 1 };

	ark::math::Quat<T, S> q1;
	ark::math::Quat<T, S> q2;
	ark::math::Quat<T, S> qr;
};

#define TI(EL, ISA) struct EL ## ISA { using T = EL; using S = ark::hal::simd::ISA; }

TI(float, None);
TI(float, Sse);
TI(float, Sse2);
TI(double, None);
TI(double, Sse);
TI(double, Sse2);

using QuatTypes = ::testing::Types
	<
	floatNone, doubleNone,
	floatSse, doubleSse,
	floatSse2, doubleSse2
	>;
TYPED_TEST_SUITE(QuatUnitTest, QuatTypes);


/*========================================================================
  Tests
========================================================================*/

/*------------------------------------------------------------------------
  Quat Constructors
------------------------------------------------------------------------*/
TYPED_TEST(QuatUnitTest, DefaultConstructor)
{
	ark::math::Quat<typename TypeParam::T, typename TypeParam::S> q;
	q;
	SUCCEED();
}

TYPED_TEST(QuatUnitTest, ElementConstructor)
{
	ark::math::Quat<typename TypeParam::T, typename TypeParam::S> q(3, 5, 7, 11);

	EXPECT_EQ(q.w(), 3) << "The w element";
	EXPECT_EQ(q.x(), 5) << "The x element";
	EXPECT_EQ(q.y(), 7) << "The y element";
	EXPECT_EQ(q.z(), 11) << "The z element";
}


/*------------------------------------------------------------------------
  Quat Unary Functions
------------------------------------------------------------------------*/

TYPED_TEST(QuatUnitTest, Negate)
{
	// When
	this->qr = -this->q1;

	// Then
	EXPECT_EQ(this->qr.w(), -3);
	EXPECT_EQ(this->qr.x(), -13);
	EXPECT_EQ(this->qr.y(), -7);
	EXPECT_EQ(this->qr.z(), -19);
}


TYPED_TEST(QuatUnitTest, Conjugate)
{
	// When
	this->qr = *this->q1;

	// Then
	EXPECT_EQ(this->qr.w(), 3);
	EXPECT_EQ(this->qr.x(), -13);
	EXPECT_EQ(this->qr.y(), -7);
	EXPECT_EQ(this->qr.z(), -19);
}


TYPED_TEST(QuatUnitTest, Inverse)
{
	// When
	this->qr = Inverse(this->q1);

	// Then
	EXPECT_NEAR(this->qr.w(), 0.005102, 0.00001);
	EXPECT_NEAR(this->qr.x(), -0.0221088, 0.00001);
	EXPECT_NEAR(this->qr.y(), -0.0119048, 0.00001);
	EXPECT_NEAR(this->qr.z(), -0.0323129, 0.00001);
}


/*------------------------------------------------------------------------
  Quat Binary Functions
------------------------------------------------------------------------*/

TYPED_TEST(QuatUnitTest, Addition)
{
	// When
	this->qr = this->q1 + this->q2;

	// Then
	EXPECT_EQ(this->qr.w(), 8);
	EXPECT_EQ(this->qr.x(), 24);
	EXPECT_EQ(this->qr.y(), 30);
	EXPECT_EQ(this->qr.z(), 48);
}


TYPED_TEST(QuatUnitTest, Subtraction)
{
	// When
	this->qr = this->q2 - this->q1;

	// Then
	EXPECT_EQ(this->qr.w(), 2);
	EXPECT_EQ(this->qr.x(), -2);
	EXPECT_EQ(this->qr.y(), 16);
	EXPECT_EQ(this->qr.z(), 10);
}


TYPED_TEST(QuatUnitTest, ScalarQuaternionMultiplication)
{
	// When
	this->qr = 5 * this->q1;

	// Then
	EXPECT_EQ(this->qr.w(), 15);
	EXPECT_EQ(this->qr.x(), 65);
	EXPECT_EQ(this->qr.y(), 35);
	EXPECT_EQ(this->qr.z(), 95);
}


TYPED_TEST(QuatUnitTest, QuaternionScalarMultiplication)
{
	// When
	this->qr = this->q1 * 3;

	// Then
	EXPECT_EQ(this->qr.w(), 9);
	EXPECT_EQ(this->qr.x(), 39);
	EXPECT_EQ(this->qr.y(), 21);
	EXPECT_EQ(this->qr.z(), 57);
}


TYPED_TEST(QuatUnitTest, QuaternionScalarDivision)
{
	// When
	this->qr = this->q1 / 2;

	// Then
	EXPECT_EQ(this->qr.w(), 1.5);
	EXPECT_EQ(this->qr.x(), 6.5);
	EXPECT_EQ(this->qr.y(), 3.5);
	EXPECT_EQ(this->qr.z(), 9.5);
}


TYPED_TEST(QuatUnitTest, I_x_I_eq_MinusOne)
{
	// When
	this->qr = this->qi * this->qi;

	// Then
	EXPECT_EQ(this->qr, -this->q_one);
}


TYPED_TEST(QuatUnitTest, J_x_J_eq_MinusOne)
{
	// When
	this->qr = this->qj * this->qj;

	// Then
	EXPECT_EQ(this->qr, -this->q_one);
}


TYPED_TEST(QuatUnitTest, K_x_K_eq_MinusOne)
{
	// When
	this->qr = this->qk * this->qk;

	// Then
	EXPECT_EQ(this->qr, -this->q_one);
}


TYPED_TEST(QuatUnitTest, I_x_J_eq_K)
{
	// When
	this->qr = this->qi * this->qj;

	// Then
	EXPECT_EQ(this->qr, this->qk);
}


TYPED_TEST(QuatUnitTest, J_x_K_eq_I)
{
	// When
	this->qr = this->qj * this->qk;

	// Then
	EXPECT_EQ(this->qr, this->qi);
}


TYPED_TEST(QuatUnitTest, K_x_I_eq_J)
{
	// When
	this->qr = this->qk * this->qi;

	// Then
	EXPECT_EQ(this->qr, this->qj);
}


TYPED_TEST(QuatUnitTest, J_x_I_eq_MinusK)
{
	// When
	this->qr = this->qj * this->qi;

	// Then
	EXPECT_EQ(this->qr, -this->qk);
}


TYPED_TEST(QuatUnitTest, K_x_J_eq_MinusI)
{
	// When
	this->qr = this->qk * this->qj;

	// Then
	EXPECT_EQ(this->qr, -this->qi);
}


TYPED_TEST(QuatUnitTest, I_x_K_eq_MinusJ)
{
	// When
	this->qr = this->qi * this->qk;

	// Then
	EXPECT_EQ(this->qr, -this->qj);
}


TYPED_TEST(QuatUnitTest, MultiplyLeftInverse_eq_1)
{
	// When
	this->qr = Inverse(this->q1) * this->q1;

	// Then
	EXPECT_NEAR(this->qr.w(), 1, 0.00001);
	EXPECT_NEAR(this->qr.x(), 0, 0.00001);
	EXPECT_NEAR(this->qr.y(), 0, 0.00001);
	EXPECT_NEAR(this->qr.z(), 0, 0.00001);
}


TYPED_TEST(QuatUnitTest, MultiplyRightInverse_eq_1)
{
	// When
	this->qr = this->q1 * Inverse(this->q1);

	// Then
	EXPECT_NEAR(this->qr.w(), 1, 0.00001);
	EXPECT_NEAR(this->qr.x(), 0, 0.00001);
	EXPECT_NEAR(this->qr.y(), 0, 0.00001);
	EXPECT_NEAR(this->qr.z(), 0, 0.00001);
}


TYPED_TEST(QuatUnitTest, Division)
{
	// When
	this->qr = this->q1 / this->q1;

	// Then
	EXPECT_NEAR(this->qr.w(), 1, 0.00001);
	EXPECT_NEAR(this->qr.x(), 0, 0.00001);
	EXPECT_NEAR(this->qr.y(), 0, 0.00001);
	EXPECT_NEAR(this->qr.z(), 0, 0.00001);
}
