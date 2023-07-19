/*************************************************************************
 * @file
 * @brief Quat Class Unit Tests
 * 
 * @details Unit tests for the Quat class implementing the Quaternion 
 * concept. As most functions are implemented against the concept, they 
 * need not be tested hwere. What's important is testing Quat's actual 
 * member functions. Consequently, as long as the constructors and 
 * accessors pass the tests, the Quat is guaranteed to work in with the 
 * quaternion expression trees (assuming the expression trees are passing 
 * all tests).
 * 
 * Having said that, the assertion is only valid for the base template. 
 * All specializations require testing of their specialized 
 * functionality. In addition to the constructors and accessors, tests 
 * must be run against any specialized functions; therefore, there is 
 * a complete suite of tests here intended to exercise specializations.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Quat.h"

#include "Cfg.h"


//************************************************************************
//  Test Suite
//************************************************************************

namespace ark::math::test::quat_unit_tests
{
	using namespace ::ark::hal;
	using namespace ::ark::math;

	/*********************************************************************
	 * @brief Quat Unit Parametric Tests Fixture
	 * 
	 * @details The parametric fixture is designed to support running 
	 * typed tests, so a single set of test source code may run tests 
	 * of multiple Quat specializations.
	 ********************************************************************/
	template<typename C>
	class QuatUnitTest : public testing::Test
	{
	protected:
		/// @brief Type of tests' quaternion scalar components
		using Scalar = typename C::Scalar;
		/// @brief the instruction set architecture to test
		using Isa = typename C::Isa;
		/// @brief Set up a standard pair of quaternions for tests
		void SetUp() override
		{
			q1 = Quat<Scalar, Isa>(3, 13, 7, 19);
			q2 = Quat<Scalar, Isa>(5, 11, 23, 29);
		}
		
		/// @name Quaternion Constants
		/// @{
		/// @brief The quaternion representing 1
		const Quat<Scalar, Isa> q_one{ 1, 0, 0, 0 };
		/// @brief The quaternion representing i
		const Quat<Scalar, Isa> qi{ 0, 1, 0, 0 };
		/// @brief The quaternion representing j
		const Quat<Scalar, Isa> qj{ 0, 0, 1, 0 };
		/// @brief The quaternion representing k
		const Quat<Scalar, Isa> qk{ 0, 0, 0, 1 };
		/// @}

		/// @name Standard Test Operands
		/// @{
		Quat<Scalar, Isa> q1;
		Quat<Scalar, Isa> q2;
		Quat<Scalar, Isa> qr;
		/// @}
	};

	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(QuatUnitTest, SseTypes);


	/**
	 * @brief Expect Quat equality
	 * 
	 * @details Google Test macro to check to see if one quaternion is 
	 * equal to another. By definition, two quaternions are equal if and 
	 * only if their respective components are equal.
	 */
	#define EXPECT_QUAT_EQ(a, b)	\
		EXPECT_EQ(a.w(), b.w()); \
		EXPECT_EQ(a.x(), b.x()); \
		EXPECT_EQ(a.y(), b.y()); \
		EXPECT_EQ(a.z(), b.z())

	/**
	 * @brief Expect Quat near-equality
	 * 
	 * @details Google Test macro to check to see if one quaternion is 
	 * nearly-equal to another. By definition, two quaternions are nearly 
	 * equal if and only if their respective components are nearly-equal.
	 */
	#define EXPECT_QUAT_NEAR(a, b, d)	\
		EXPECT_NEAR(a.w(), b.w(), d); \
		EXPECT_NEAR(a.x(), b.x(), d); \
		EXPECT_NEAR(a.y(), b.y(), d); \
		EXPECT_NEAR(a.z(), b.z(), d)


	//************************************************************************
	// Tests
	//************************************************************************

	//========================================================================
	// Quat Constructors
	//========================================================================

	TYPED_TEST(QuatUnitTest, DefaultConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		ark::math::Quat<Scalar, Isa> q;
		q;

		// Then
		SUCCEED();
	}

	TYPED_TEST(QuatUnitTest, ElementConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		ark::math::Quat<Scalar, Isa> q(3, 5, 7, 11);
		q;

		// Then
		EXPECT_EQ(q.w(), 3) << "The w element";
		EXPECT_EQ(q.x(), 5) << "The x element";
		EXPECT_EQ(q.y(), 7) << "The y element";
		EXPECT_EQ(q.z(), 11) << "The z element";
	}


	//========================================================================
	// Quat Unary Functions
	//========================================================================


	TYPED_TEST(QuatUnitTest, NegationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = -this->q1;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
	}


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


	TYPED_TEST(QuatUnitTest, ConjugateReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = *this->q1;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	TYPED_TEST(QuatUnitTest, DotProduct)
	{
		// When
		auto r = Dot(this->q1, this->q2);

		// Then
		EXPECT_EQ(r, 870);
	}


	TYPED_TEST(QuatUnitTest, InverseReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = Inverse(this->q1);

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	//========================================================================
	// Quat Binary Functions
	//========================================================================

	TYPED_TEST(QuatUnitTest, AdditionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 + this->q2;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
	}


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


	TYPED_TEST(QuatUnitTest, SubtractionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 - this->q2;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	TYPED_TEST(QuatUnitTest, ScalarQuaternionMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};
		
		// When
		auto result = scalar * this->q2;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	TYPED_TEST(QuatUnitTest, QuaternionScalarMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};

		// When
		auto result = this->q2 * scalar;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	TYPED_TEST(QuatUnitTest, ScalarDivisionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};

		// When
		auto result = this->q2 / scalar;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
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


	TYPED_TEST(QuatUnitTest, MultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 * this->q2;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;

		if (!(isSame ^ isBasicExpression))
		{
			auto r2 = this->q1 * this->q2;
			auto r3 = this->q1 * this->q1;
		}

		EXPECT_TRUE(isSame ^ isBasicExpression);
	}


	TYPED_TEST(QuatUnitTest, I_x_I_eq_MinusOne)
	{
		// When
		this->qr = this->qi * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QuatUnitTest, J_x_J_eq_MinusOne)
	{
		// When
		this->qr = this->qj * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QuatUnitTest, K_x_K_eq_MinusOne)
	{
		// When
		this->qr = this->qk * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QuatUnitTest, I_x_J_eq_K)
	{
		// When
		this->qr = this->qi * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qk);
	}


	TYPED_TEST(QuatUnitTest, J_x_K_eq_I)
	{
		// When
		this->qr = this->qj * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qi);
	}


	TYPED_TEST(QuatUnitTest, K_x_I_eq_J)
	{
		// When
		this->qr = this->qk * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qj);
	}


	TYPED_TEST(QuatUnitTest, J_x_I_eq_MinusK)
	{
		// When
		this->qr = this->qj * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qk);
	}


	TYPED_TEST(QuatUnitTest, K_x_J_eq_MinusI)
	{
		// When
		this->qr = this->qk * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qi);
	}


	TYPED_TEST(QuatUnitTest, I_x_K_eq_MinusJ)
	{
		// When
		this->qr = this->qi * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qj);
	}


	TYPED_TEST(QuatUnitTest, MultiplyLeftInverse_eq_1)
	{
		// When
		this->qr = Inverse(this->q1) * this->q1;

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}


	TYPED_TEST(QuatUnitTest, MultiplyRightInverse_eq_1)
	{
		// When
		this->qr = this->q1 * Inverse(this->q1);

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}


	TYPED_TEST(QuatUnitTest, DivisionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 / this->q2;

		// Then
		bool isBasicExpression = std::is_base_of_v<QuaternionExpr, decltype(result)>
			&& std::is_same_v<QuatBasic<typename TypeParam::Scalar>, decltype(this->q1)>;
		bool isSame = std::is_same_v<decltype(this->q1), decltype(result)>;
		EXPECT_TRUE(isSame ^ isBasicExpression);
	}


	TYPED_TEST(QuatUnitTest, Division)
	{
		// When
		this->qr = this->q1 / this->q1;

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}

}
