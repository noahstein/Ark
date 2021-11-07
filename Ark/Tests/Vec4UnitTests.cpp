/*************************************************************************
 * @file
 * @brief 4D Vec Class Specializations Unit Tests
 * 
 * @details Unit tests for the 4-D Vec class specializations. This file 
 * implements a full suite of tests for 4-D vectors, likely the dimension 
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
	 * @brief 4-D Vec Unit Parametric Tests Fixture
	 * 
	 * @details The parametric fixture is designed to support running 
	 * typed tests, so a single set of test source code may run tests 
	 * of multiple 4-D Vec specializations.
	 ********************************************************************/
	template<typename C>
	class Vec4UnitTest : public testing::Test
	{
	protected:
		/// @brief Type of tests' quaternion scalar components
		using Scalar = typename C::Scalar;
		/// @brief the instruction set architecture to test
		using Isa = typename C::Isa;
		/// @brief Set up a standard pair of quaternions for tests
		void SetUp() override
		{
			v1 = Vec<Scalar, 4, Isa>(3, 13, 7, 19);
			v2 = Vec<Scalar, 4, Isa>(5, 11, 23, 29);
		}

		/// @name Standard Test Operands
		/// @{
		Vec<Scalar, 4, Isa> v1;
		Vec<Scalar, 4, Isa> v2;
		Vec<Scalar, 4, Isa> vr;
		/// @}
	};


	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(Vec4UnitTest, SseTypes);


	//************************************************************************
	// Tests
	//************************************************************************

	//========================================================================
	// Constructors and Fundamental Member Functions
	//========================================================================

	TYPED_TEST(Vec4UnitTest, DefaultConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		auto v = Vec<Scalar, 4, Isa>{};
		v;

		// Then
		SUCCEED();
	}


	TYPED_TEST(Vec4UnitTest, ComponentConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		auto v = Vec<Scalar, 4, Isa>{2, 3, 5, 7};
		v;

		// Then
		EXPECT_EQ(v(0), 2);
		EXPECT_EQ(v(1), 3);
		EXPECT_EQ(v(2), 5);
		EXPECT_EQ(v(3), 7);
	}


	TYPED_TEST(Vec4UnitTest, CopyConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v1 = Vec<Scalar, 4, Isa>{2, 3, 5, 7};

		// When
		auto v{v1};

		// Then
		EXPECT_EQ(v(0), 2);
		EXPECT_EQ(v(1), 3);
		EXPECT_EQ(v(2), 5);
		EXPECT_EQ(v(3), 7);
	}


	TYPED_TEST(Vec4UnitTest, OperatorEqual)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v1 = Vec<Scalar, 4, Isa>{2, 3, 5, 7};
		auto v2 = Vec<Scalar, 4, Isa>{10, 11, 12, 13};
		auto v3 = Vec<Scalar, 4, Isa>{};

		// When
		v3 = v2 = v1;

		// Then
		EXPECT_EQ(v2(0), 2);
		EXPECT_EQ(v2(1), 3);
		EXPECT_EQ(v2(2), 5);
		EXPECT_EQ(v2(3), 7);

		EXPECT_EQ(v3(0), 2);
		EXPECT_EQ(v3(1), 3);
		EXPECT_EQ(v3(2), 5);
		EXPECT_EQ(v3(3), 7);
	}


	TYPED_TEST(Vec4UnitTest, AccessorAt0)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{1, 2, 3, 5};

		// When
		auto v_0 = v(0);

		// Then
		EXPECT_EQ(v_0, 1);
	}


	TYPED_TEST(Vec4UnitTest, AccessorAt1)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{1, 2, 3, 5};

		// When
		auto v_1 = v(1);

		// Then
		EXPECT_EQ(v_1, 2);
	}


	TYPED_TEST(Vec4UnitTest, AccessorAt2)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{1, 2, 3, 5};

		// When
		auto v_2 = v(2);

		// Then
		EXPECT_EQ(v_2, 3);
	}


	TYPED_TEST(Vec4UnitTest, AccessorAt3)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{1, 2, 3, 5};

		// When
		auto v_3 = v(3);

		// Then
		EXPECT_EQ(v_3, 5);
	}


	//========================================================================
	// Mathematical Functions
	//========================================================================

	TYPED_TEST(Vec4UnitTest, Negate)
	{
		// When
		this->vr = -this->v1;

		// Then
		EXPECT_EQ(this->vr(0), -3);
		EXPECT_EQ(this->vr(1), -13);
		EXPECT_EQ(this->vr(2), -7);
		EXPECT_EQ(this->vr(3), -19);
	}


	TYPED_TEST(Vec4UnitTest, Add)
	{
		// When
		this->vr = this->v1 + this->v2;

		// Then
		EXPECT_EQ(this->vr(0), 8);
		EXPECT_EQ(this->vr(1), 24);
		EXPECT_EQ(this->vr(2), 30);
		EXPECT_EQ(this->vr(3), 48);
	}


	TYPED_TEST(Vec4UnitTest, Subtract)
	{
		// When
		this->vr = this->v1 - this->v2;

		// Then
		EXPECT_EQ(this->vr(0), -2);
		EXPECT_EQ(this->vr(1), 2);
		EXPECT_EQ(this->vr(2), -16);
		EXPECT_EQ(this->vr(3), -10);
	}


	TYPED_TEST(Vec4UnitTest, ScalarVectorMultiply)
	{
		// When
		this->vr = this->v1 * 10;

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 70);
		EXPECT_EQ(this->vr(3), 190);
	}


	TYPED_TEST(Vec4UnitTest, VectorScalarMultiply)
	{
		// When
		this->vr = 10 * this->v1;

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 70);
		EXPECT_EQ(this->vr(3), 190);
	}


	TYPED_TEST(Vec4UnitTest, VectorScalarDivide)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{4, 6, 10, 14};

		// When
		this->vr = v / 2;

		// Then
		EXPECT_EQ(this->vr(0), 2);
		EXPECT_EQ(this->vr(1), 3);
		EXPECT_EQ(this->vr(2), 5);
		EXPECT_EQ(this->vr(3), 7);
	}


	TYPED_TEST(Vec4UnitTest, EqualityCheckSame)
	{
		// When
		bool result = this->v1 == this->v1;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(Vec4UnitTest, EqualityCheckDifferent)
	{
		// When
		bool result = this->v1 == this->v2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(Vec4UnitTest, InequalityCheckSame)
	{
		// When
		bool result = this->v1 != this->v1;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(Vec4UnitTest, InequalityCheckDifferent)
	{
		// When
		bool result = this->v1 != this->v2;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(Vec4UnitTest, VectorDotProduct)
	{
		// When
		float result = Dot(this->v1, this->v2);

		// Then
		EXPECT_EQ(result, 870);
	}


	TYPED_TEST(Vec4UnitTest, VectorCrossProduct4D)
	{
		// When
		this->vr = Cross(this->v1, this->v2);

		// Then
		EXPECT_EQ(this->vr(0), 222);
		EXPECT_EQ(this->vr(1), -34);
		EXPECT_EQ(this->vr(2), -32);
		EXPECT_EQ(this->vr(3), 0);
	}


	TYPED_TEST(Vec4UnitTest, VectorNorm)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{2, 4, 1, 2};

		// When
		float result = Norm(v);

		// Then
		EXPECT_EQ(result, 5);
	}
}
