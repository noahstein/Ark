/*========================================================================
Description
	Unit tests for the Vec class implementing the Vector concept. 
	As most functions are implemented against the concept, they need not 
	be tested hwere. What's important is testing Vec's actual member 
	functions.
	
	The other important aspect is to test any functionality that is 
	overridden in specializations of the class. For example, Vec<float> 
	has platform-specific specializations to use SIMD hardware; 
	therefore, there are tests here to ensure those work properly.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/


/*========================================================================
 Dependencies
========================================================================*/
#include "gtest/gtest.h"
#include "Ark/Math/Vec.h"

template<typename S> using Vec3 = ark::math::Vec<S, 3>;
template<typename S> using Vec4 = ark::math::Vec<S, 4>;


/*========================================================================
 Fixtures
========================================================================*/

/*------------------------------------------------------------------------
 Vec<float> Test Fixture
------------------------------------------------------------------------*/
class VecFloatUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		v1 = Vec3<float>{3.2f, 13.5f, 7.3f};
		v2 = Vec3<float>{5, 11, 23};
	}
	Vec3<float> v1;
	Vec3<float> v2;
	Vec3<float> vr;
};


/*========================================================================
 Tests
========================================================================*/

TEST(VecUnitTests, DefaultConstructor)
{
	Vec3<int> v;
	SUCCEED();
}


TEST(VecUnitTests, ElementConstructor)
{
	// When
	auto v = Vec3<float>{2.0f, 3.0f, 5.0f};

	// Then
	EXPECT_EQ(v(0), 2.0f);
	EXPECT_EQ(v(1), 3.0f);
	EXPECT_EQ(v(2), 5.0f);
}


TEST(VecUnitTests, VectorCopyConstructor)
{
	// Given
	auto v1 = Vec3<float>{2.0f, 3.0f, 5.0f};

	// When
	Vec3<int> v2{v1};

	// Then
	EXPECT_EQ(v2(0), 2);
	EXPECT_EQ(v2(1), 3);
	EXPECT_EQ(v2(2), 5);
}


TEST(VecUnitTests, OperatorEqual)
{
	// Given
	auto v1 = Vec3<float>{2.0f, 3.0f, 5.0f};
	ark::math::Vec<int, 3> v2{10, 11, 12};
	ark::math::Vec<int, 3> v3;

	// When
	v3 = v2 = v1;

	// Then
	EXPECT_EQ(v2(0), 2);
	EXPECT_EQ(v2(1), 3);
	EXPECT_EQ(v2(2), 5);

	EXPECT_EQ(v3(0), 2);
	EXPECT_EQ(v3(1), 3);
	EXPECT_EQ(v3(2), 5);
}


TEST(VecUnitTests, Accessor)
{
	// When
	Vec3<float> v{1.0f, 2.0f, 3.0f};

	// Then
	EXPECT_EQ(v(0), 1.0f);
	EXPECT_EQ(v(1), 2.0f);
	EXPECT_EQ(v(2), 3.0f);
}


/*------------------------------------------------------------------------
 4-dimensional Tests (for SIMD specializations)
------------------------------------------------------------------------*/

TEST(VecUnitTests, ElementConstructor4D)
{
	// When
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// Then
	EXPECT_EQ(v(0), 2.0f);
	EXPECT_EQ(v(1), 3.0f);
	EXPECT_EQ(v(2), 5.0f);
	EXPECT_EQ(v(3), 7.0f);
}


TEST(VecUnitTests, GeneralVecConstructor4D)
{
	// Given
	auto s = Vec4<int>{2, 3, 5, 7};

	// When
	auto v = s;

	// Then
	EXPECT_EQ(v(0), 2.0f);
	EXPECT_EQ(v(1), 3.0f);
	EXPECT_EQ(v(2), 5.0f);
	EXPECT_EQ(v(3), 7.0f);
}

TEST(VecUnitTests, Negate)
{
	// Given
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// When
	auto vr = -v;

	// Then
	EXPECT_EQ(vr(0), -2.0f);
	EXPECT_EQ(vr(1), -3.0f);
	EXPECT_EQ(vr(2), -5.0f);
	EXPECT_EQ(vr(3), -7.0f);
}

TEST(VecUnitTests, Add)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{23.0f, 29.0f, 31.0f, 37.0f};

	// When
	auto vr = v1 + v2;

	// Then
	EXPECT_EQ(vr(0), 34.0f);
	EXPECT_EQ(vr(1), 42.0f);
	EXPECT_EQ(vr(2), 48.0f);
	EXPECT_EQ(vr(3), 56.0f);
}

TEST(VecUnitTests, Subtract)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{23.0f, 29.0f, 31.0f, 37.0f};

	// When
	auto vr = v1 - v2;

	// Then
	EXPECT_EQ(vr(0), -12.0f);
	EXPECT_EQ(vr(1), -16.0f);
	EXPECT_EQ(vr(2), -14.0f);
	EXPECT_EQ(vr(3), -18.0f);
}

TEST(VecUnitTests, ScalarVectorMultiply)
{
	// Given
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// When
	auto vr = v * 10.0f;

	// Then
	EXPECT_EQ(vr(0), 20.0f);
	EXPECT_EQ(vr(1), 30.0f);
	EXPECT_EQ(vr(2), 50.0f);
	EXPECT_EQ(vr(3), 70.0f);
}

TEST(VecUnitTests, VectorScalarMultiply)
{
	// Given
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// When
	auto vr = 10.0f * v;

	// Then
	EXPECT_EQ(vr(0), 20.0f);
	EXPECT_EQ(vr(1), 30.0f);
	EXPECT_EQ(vr(2), 50.0f);
	EXPECT_EQ(vr(3), 70.0f);
}

TEST(VecUnitTests, VectorScalarDivide)
{
	// Given
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// When
	auto vr = v / 2.0f;

	// Then
	EXPECT_EQ(vr(0), 1.0f);
	EXPECT_EQ(vr(1), 1.5f);
	EXPECT_EQ(vr(2), 2.5f);
	EXPECT_EQ(vr(3), 3.5f);
}

TEST(VecUnitTests, EqualityCheckSame)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};

	// When
	bool result = v1 == v2;

	// Then
	EXPECT_TRUE(result);
}

TEST(VecUnitTests, EqualityCheckDifferent)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{23.0f, 29.0f, 31.0f, 37.0f};

	// When
	bool result = v1 == v2;

	// Then
	EXPECT_FALSE(result);
}

TEST(VecUnitTests, InequalityCheckSame)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};

	// When
	bool result = v1 != v2;

	// Then
	EXPECT_FALSE(result);
}

TEST(VecUnitTests, InequalityCheckDifferent)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{11.0f, 13.0f, 17.0f, 37.0f};

	// When
	bool result = v1 != v2;

	// Then
	EXPECT_TRUE(result);
}

TEST(VecUnitTests, VectorDotProduct)
{
	// Given
	auto v1 = Vec4<float>{11.0f, 13.0f, 17.0f, 19.0f};
	auto v2 = Vec4<float>{23.0f, 29.0f, 31.0f, 37.0f};

	// When
	float result = Dot(v1, v2);

	// Then
	EXPECT_EQ(result, 1860);
}


TEST(VecUnitTests, VectorCrossProduct4D)
{
	// Given
	auto v1 = Vec4<float>{2.0f, 3.0f, 5.0f, 0.0f};
	auto v2 = Vec4<float>{7.0f, 11.0f, 13.0f, 0.0f};

	// When
	Vec4<float> result = Cross(v1, v2);

	// Then
	EXPECT_EQ(result(0), -16.0f);
	EXPECT_EQ(result(1), 9.0f);
	EXPECT_EQ(result(2), 1.0f);
	EXPECT_EQ(result(3), 0.0f);
}

TEST(VecUnitTests, VectorNorm)
{
	// Given
	auto v = Vec4<float>{2.0f, 3.0f, 5.0f, 7.0f};

	// When
	float result = Norm(v);

	// Then
	EXPECT_EQ(result, std::sqrt(87.0f));
}
