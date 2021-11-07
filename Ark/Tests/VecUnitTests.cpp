/*************************************************************************
 * @file
 * @brief Vec Class Unit Tests
 * 
 * @details Unit tests for the Vec class implementing the Vector 
 * concept. As most functions are implemented against the concept, they 
 * need not be tested hwere. What's important is testing Vec's actual 
 * member functions. Consequently, as long as the constructors, `Size()`, 
 * and accessors pass the tests, the Vec is guaranteed to work in with 
 * the vector expression trees (assuming the expression trees are passing 
 * all tests).
 * 
 * Having said that, the assertion is only valid for the base template. 
 * All specializations require testing of their specialized 
 * functionality. In addition to the aformentioned functions, tests must 
 * be run against any specialized functions; therefore, there is a 
 * complete suite of tests here intended to exercise specializations.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Vec.h"

#include "Cfg.h"

//************************************************************************
//  Test Suite
//************************************************************************

namespace ark::math::test::vec_unit_tests
{
	using namespace ::ark::hal;
	using namespace ::ark::math;


	/*********************************************************************
	 * @brief Vec Unit Parametric Tests Fixture
	 * 
	 * @details The unit test fixture is designed with the focus around 
	 * testing 3-D vectors. There are some 2-D tests. There is a dearth 
	 * of 4-D tests as they have a low priority due to the presence of 
	 * tests for SIMD specializations.
	 ********************************************************************/
	template<typename C>
	class VecUnitTest : public testing::Test
	{
	protected:
		/**
		 * @brief The scalar type of the vector to undergo testing
		 */
		using Scalar = typename C::Scalar;

		void SetUp() override
		{
			v1 = Vec<Scalar, 3>{3, 13, 7};
	        v2 = Vec<Scalar, 3>{5, 11, 23};
		}

		Vec<Scalar, 3> v1;
		Vec<Scalar, 3> v2;
		Vec<Scalar, 3> vr;
	};


	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(VecUnitTest, StdTypes);


	//************************************************************************
	//  Tests
	//************************************************************************

	TYPED_TEST(VecUnitTest, DefaultConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		auto v = Vec<Scalar, 3>{};
		v;

		// Then
		SUCCEED();
	}


	TYPED_TEST(VecUnitTest, ElementConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		auto v = Vec<Scalar, 3>{2, 3, 5};

		// Then
		EXPECT_EQ(v(0), 2);
		EXPECT_EQ(v(1), 3);
		EXPECT_EQ(v(2), 5);
	}


	TYPED_TEST(VecUnitTest, VectorCopyConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v1 = Vec<Scalar, 3>{2, 3, 5};

		// When
		auto v2 = Vec<Scalar, 3>{v1};

		// Then
		EXPECT_EQ(v2(0), 2);
		EXPECT_EQ(v2(1), 3);
		EXPECT_EQ(v2(2), 5);
	}


	TYPED_TEST(VecUnitTest, OperatorEqual)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v1 = Vec<Scalar, 3>{2, 3, 5};
		auto v2 = Vec<Scalar, 3>{10, 11, 12};
		auto v3 = Vec<Scalar, 3>{};

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


	TYPED_TEST(VecUnitTest, AccessorAt0)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 3>{1, 2, 3};

		// When
		auto v_0 = v(0);

		// Then
		EXPECT_EQ(v_0, 1);
	}


	TYPED_TEST(VecUnitTest, AccessorAt1)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 3>{1, 2, 3};

		// When
		auto v_1 = v(1);

		// Then
		EXPECT_EQ(v_1, 2);
	}


	TYPED_TEST(VecUnitTest, AccessorAt2)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 3>{1, 2, 3};

		// When
		auto v_2 = v(2);

		// Then
		EXPECT_EQ(v_2, 3);
	}
}
