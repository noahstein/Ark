/*========================================================================
Description
	Unit tests for the Mtx class implementing the Matrix concept. 
	As most functions are implemented against the concept, they need not 
	be tested hwere. What's important is testing Mtx's actual member 
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
#include "Ark/Math/Mtx.h"

template<typename S> using Mtx = ark::math::Mtx<S, 2, 2>;


/*========================================================================
 Fixtures
========================================================================*/

/*------------------------------------------------------------------------
 Mtx<float> Test Fixture
------------------------------------------------------------------------*/
class MtxFloatUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		m1 = Mtx<float>({3.2f, 13.5f, 7.3f});
		m2 = Mtx<float>({5, 11, 23});
	}
	Mtx<float> m1;
	Mtx<float> m2;
	Mtx<float> mr;
};


/*========================================================================
 Tests
========================================================================*/

TEST(MtxUnitTests, DefaultConstructor)
{
	Mtx<int> v;
	SUCCEED();
}

TEST(MtxUnitTests, Accessor)
{
	// When
	Mtx<float> m({2.0f, 3.0f,
	              5.0f, 7.0f});

	// Then
	EXPECT_EQ(m(0, 0), 2.0f);
	EXPECT_EQ(m(0, 1), 3.0f);
	EXPECT_EQ(m(1, 0), 5.0f);
	EXPECT_EQ(m(1, 1), 7.0f);
}

