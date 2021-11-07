/*************************************************************************
 * @file
 * @brief 2D Vec Class Specializations Unit Tests
 * 
 * @details Unit tests for the 2-D Vec class specializations. This file 
 * implements a full suite of tests for 2-D vectors, likely the dimension 
 * most-specialized for all the platforms.
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
	 * @brief 2-D Vec Unit Parametric Tests Fixture
	 * 
	 * @details The parametric fixture is designed to support running 
	 * typed tests, so a single set of test source code may run tests 
	 * of multiple 2-D Vec specializations.
	 ********************************************************************/
	template<typename C>
	class Vec2UnitTest : public testing::Test
	{
	protected:
		/// @brief Type of tests' quaternion scalar components
		using Scalar = typename C::Scalar;

		/// @brief the instruction set architecture to test
		using Isa = typename C::Isa;

		/// @brief Set up a standard pair of quaternions for tests
		void SetUp() override
		{
			v1 = Vec<Scalar, 2, Isa>(3, 13);
			v2 = Vec<Scalar, 2, Isa>(5, 11);
		}

		/// @name Standard Test Operands
		/// @{
		Vec<Scalar, 2, Isa> v1;
		Vec<Scalar, 2, Isa> v2;
		Vec<Scalar, 2, Isa> vr;
		/// @}
	};


	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(Vec2UnitTest, SseTypes);


	//************************************************************************
	// Tests
	//************************************************************************

	//========================================================================
	// Constructors and Fundamental Member Functions
	//========================================================================

	TYPED_TEST(Vec2UnitTest, DefaultConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		auto v = Vec<Scalar, 2, Isa>{};
		v;

		// Then
		SUCCEED();
	}


	TYPED_TEST(Vec2UnitTest, ComponentConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		auto v = Vec<Scalar, 2, Isa>{2, 3};
		v;

		// Then
		EXPECT_EQ(v(0), 2);
		EXPECT_EQ(v(1), 3);
	}


	TYPED_TEST(Vec2UnitTest, CopyConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v1 = Vec<Scalar, 2, Isa>{2, 3};

		// When
		auto v{v1};

		// Then
		EXPECT_EQ(v(0), 2);
		EXPECT_EQ(v(1), 3);
	}


	TYPED_TEST(Vec2UnitTest, OperatorEqual)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v1 = Vec<Scalar, 2, Isa>{2, 3};
		auto v2 = Vec<Scalar, 2, Isa>{10, 11};
		auto v3 = Vec<Scalar, 2, Isa>{};

		// When
		v3 = v2 = v1;

		// Then
		EXPECT_EQ(v2(0), 2);
		EXPECT_EQ(v2(1), 3);

		EXPECT_EQ(v3(0), 2);
		EXPECT_EQ(v3(1), 3);
	}


	TYPED_TEST(Vec2UnitTest, AccessorAt0)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 2, Isa>{1, 2};

		// When
		auto v_0 = v(0);

		// Then
		EXPECT_EQ(v_0, 1);
	}


	TYPED_TEST(Vec2UnitTest, AccessorAt1)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 2, Isa>{1, 2};

		// When
		auto v_1 = v(1);

		// Then
		EXPECT_EQ(v_1, 2);
	}


	//========================================================================
	// Mathematical Functions
	//========================================================================

	TYPED_TEST(Vec2UnitTest, Negate)
	{
		// When
		this->vr = -this->v1;

		// Then
		EXPECT_EQ(this->vr(0), -3);
		EXPECT_EQ(this->vr(1), -13);
	}


	TYPED_TEST(Vec2UnitTest, Add)
	{
		// When
		this->vr = this->v1 + this->v2;

		// Then
		EXPECT_EQ(this->vr(0), 8);
		EXPECT_EQ(this->vr(1), 24);
	}


	TYPED_TEST(Vec2UnitTest, Subtract)
	{
		// When
		this->vr = this->v1 - this->v2;

		// Then
		EXPECT_EQ(this->vr(0), -2);
		EXPECT_EQ(this->vr(1), 2);
	}


	TYPED_TEST(Vec2UnitTest, ScalarVectorMultiply)
	{
		// When
		this->vr = this->v1 * 10;

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
	}


	TYPED_TEST(Vec2UnitTest, VectorScalarMultiply)
	{
		// When
		this->vr = 10 * this->v1;

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
	}


	TYPED_TEST(Vec2UnitTest, VectorScalarDivide)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 2, Isa>{4, 6};

		// When
		this->vr = v / 2;

		// Then
		EXPECT_EQ(this->vr(0), 2);
		EXPECT_EQ(this->vr(1), 3);
	}


	TYPED_TEST(Vec2UnitTest, EqualityCheckSame)
	{
		// When
		bool result = this->v1 == this->v1;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(Vec2UnitTest, EqualityCheckDifferent)
	{
		// When
		bool result = this->v1 == this->v2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(Vec2UnitTest, InequalityCheckSame)
	{
		// When
		bool result = this->v1 != this->v1;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(Vec2UnitTest, InequalityCheckDifferent)
	{
		// When
		bool result = this->v1 != this->v2;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(Vec2UnitTest, VectorDotProduct)
	{
		// When
		float result = Dot(this->v1, this->v2);

		// Then
		EXPECT_EQ(result, 158);
	}


#if 1
	TYPED_TEST(Vec2UnitTest, VectorCrossProduct2D)
	{
		// When
		auto r = Cross(this->v1, this->v2);

		// Then
		EXPECT_EQ(r, -32);
	}
#endif


	TYPED_TEST(Vec2UnitTest, VectorNorm)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 2, Isa>{3, 4};

		// When
		float result = Norm(v);

		// Then
		EXPECT_EQ(result, 5);
	}
}
