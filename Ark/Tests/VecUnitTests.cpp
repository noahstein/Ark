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
	Copyright (c) 2021 Noah Stein. All Rights Reserverd.
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
		v1 = Vec<float>({3.2f, 13.5f, 7.3f});
		v2 = Vec<float>({5, 11, 23});
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

TEST(VecUnitTests, Accessor)
{
	// When
	Vec<float> v({1.0f, 2.0f, 3.0f});

	// Then
	EXPECT_EQ(v(0), 1.0f);
	EXPECT_EQ(v(1), 2.0f);
	EXPECT_EQ(v(2), 3.0f);
}

