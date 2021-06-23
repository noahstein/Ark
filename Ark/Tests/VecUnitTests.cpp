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

template<typename S> using Vec = ark::math::Vec<S, 3>;


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
		v1 = Vec<float>{3.2f, 13.5f, 7.3f};
		v2 = Vec<float>{5, 11, 23};
	}
	Vec<float> v1;
	Vec<float> v2;
	Vec<float> vr;
};


/*========================================================================
 Tests
========================================================================*/

TEST(VecUnitTests, DefaultConstructor)
{
	Vec<int> v;
	SUCCEED();
}


TEST(VecUnitTests, ElementConstructor)
{
	// When
	auto v = Vec<float>{2.0f, 3.0f, 5.0f};

	// Then
	EXPECT_EQ(v(0), 2.0f);
	EXPECT_EQ(v(1), 3.0f);
	EXPECT_EQ(v(2), 5.0f);
}


TEST(VecUnitTests, VectorCopyConstructor)
{
	// Given
	auto v1 = Vec<float>{2.0f, 3.0f, 5.0f};

	// When
	Vec<int> v2{v1};

	// Then
	EXPECT_EQ(v2(0), 2);
	EXPECT_EQ(v2(1), 3);
	EXPECT_EQ(v2(2), 5);
}


TEST(VecUnitTests, OperatorEqual)
{
	// Given
	auto v1 = Vec<float>{2.0f, 3.0f, 5.0f};
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
	Vec<float> v{1.0f, 2.0f, 3.0f};

	// Then
	EXPECT_EQ(v(0), 1.0f);
	EXPECT_EQ(v(1), 2.0f);
	EXPECT_EQ(v(2), 3.0f);
}
