/*************************************************************************
 * @file
 * @brief Quat Unit Tests Template
 * 
 * @details The basic Quat class and all optimized implementation classes 
 * have the same testing needs. Thus, the tests are defined in this header 
 * file that gets used by each implementation's unit test cpp file.
 * The test class in this file relies upon preprocessor macros to be 
 * properly compiled. As all tests end up in the same executable, each 
 * class's tests need to be in unique classes. Additionally, the Visual 
 * Studio 2022 Test Explorer (as of August 2023) fails to run tests if 
 * the test classes have the same name in different namespaces. Thus, 
 * this file has three requirements be met in the unit test cpp file 
 * in order to be properly included:
 * 
 * # QUAT_TEST_NAMESPACE: This preprocessor macro must be defined to 
 *   specify the namespace in ark::math::test in which to place the 
 *   test class. This structures the tests better in Test Explorer.
 * 
 * # QUAT_TEST_CLASS: This preprocessor must be defined to specify 
 *   the name of the unit test class. As stated above, if a unit 
 *   test class name is reused in different namespaces, the Visual 
 *   Studio Test Explorer will have problems running the tests.
 * 
 * # QuatTestTypes: This using definition specifies an array of 
 *   Google Test types to use in the template unit test class.
 * 
 * With those configured properly, including this header file afterwards 
 * results in the definition of a class to unit test a quaternion 
 * implementation.
 * 
 * @author Noah Stein
 * @copyright © 2021-2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Quat.h"


//************************************************************************
//  Test Suite
//************************************************************************

namespace ark::math::test::QUAT_TEST_NAMESPACE
{
	/*********************************************************************
	 * @brief Quat Unit Parametric Tests Fixture
	 * 
	 * @details The parametric fixture is designed to support running 
	 * typed tests, so a single set of test source code may run tests 
	 * of multiple Quat specializations.
	 ********************************************************************/
	template<typename C>
	class QUAT_TEST_CLASS : public testing::Test
	{
	protected:
		using Scalar = typename C::Scalar;
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

		/// @name Support Functions
		/// @{

		/**
		 * @brief Predicate to determine if a result is of a basic
		 * quaternion expression type. This is useful when determining if
		 * the result value should be an optimized type.
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if basic expression, false if optimized type
		 */
		template<typename R>
		bool IsResultTypeBasicExpression(R& result) const
		{
			bool derived = std::derived_from<R, QuaternionExpr>;
			bool same = std::same_as<QuatBasic<Scalar>, Quat<Scalar, Isa>>;
			return derived && same;
		}


		/**
		 * @brief Predicate to determine if a result type the same as that
		 * under test.
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if types are same, false if different
		 */
		template<typename R>
		bool IsResultTypeSame(R& result) const
		{
			return std::same_as<R, Quat<Scalar, Isa>>;
		}


		/**
		 * @brief Predicate to determine if a result type is correctly
		 * either a basic expression or an optimized type, depending upon
		 * whether the parameters are basic or optimized.
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if types are properly optimized, false if not.
		 */
		template<typename R>
		bool IsSpecializedCorrectly(R& result) const
		{
			bool isBasicExpression = IsResultTypeBasicExpression(result);
			bool isSame = IsResultTypeSame(result);
			bool isCorrect = isBasicExpression ^ isSame;
			return isCorrect;
		}

		/// @}
	};

	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(QUAT_TEST_CLASS, QuatTestTypes);


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

	TYPED_TEST(QUAT_TEST_CLASS, DefaultConstructor)
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

	TYPED_TEST(QUAT_TEST_CLASS, ElementConstructor)
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


	TYPED_TEST(QUAT_TEST_CLASS, NegationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = -this->q1;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Negate)
	{
		// When
		this->qr = -this->q1;

		// Then
		EXPECT_EQ(this->qr.w(), -3);
		EXPECT_EQ(this->qr.x(), -13);
		EXPECT_EQ(this->qr.y(), -7);
		EXPECT_EQ(this->qr.z(), -19);
	}


	TYPED_TEST(QUAT_TEST_CLASS, ConjugateReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = *this->q1;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Conjugate)
	{
		// When
		this->qr = *this->q1;

		// Then
		EXPECT_EQ(this->qr.w(), 3);
		EXPECT_EQ(this->qr.x(), -13);
		EXPECT_EQ(this->qr.y(), -7);
		EXPECT_EQ(this->qr.z(), -19);
	}


	TYPED_TEST(QUAT_TEST_CLASS, DotProduct)
	{
		// When
		auto r = Dot(this->q1, this->q2);

		// Then
		EXPECT_EQ(r, 870);
	}


	TYPED_TEST(QUAT_TEST_CLASS, InverseReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = Inverse(this->q1);

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Inverse)
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

	TYPED_TEST(QUAT_TEST_CLASS, AdditionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 + this->q2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Addition)
	{
		// When
		this->qr = this->q1 + this->q2;

		// Then
		EXPECT_EQ(this->qr.w(), 8);
		EXPECT_EQ(this->qr.x(), 24);
		EXPECT_EQ(this->qr.y(), 30);
		EXPECT_EQ(this->qr.z(), 48);
	}


	TYPED_TEST(QUAT_TEST_CLASS, SubtractionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 - this->q2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Subtraction)
	{
		// When
		this->qr = this->q2 - this->q1;

		// Then
		EXPECT_EQ(this->qr.w(), 2);
		EXPECT_EQ(this->qr.x(), -2);
		EXPECT_EQ(this->qr.y(), 16);
		EXPECT_EQ(this->qr.z(), 10);
	}


	TYPED_TEST(QUAT_TEST_CLASS, ScalarQuaternionMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};

		// When
		auto result = scalar * this->q2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, ScalarQuaternionMultiplication)
	{
		// When
		this->qr = 5 * this->q1;

		// Then
		EXPECT_EQ(this->qr.w(), 15);
		EXPECT_EQ(this->qr.x(), 65);
		EXPECT_EQ(this->qr.y(), 35);
		EXPECT_EQ(this->qr.z(), 95);
	}


	TYPED_TEST(QUAT_TEST_CLASS, QuaternionScalarMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};

		// When
		auto result = this->q2 * scalar;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, QuaternionScalarMultiplication)
	{
		// When
		this->qr = this->q1 * 3;

		// Then
		EXPECT_EQ(this->qr.w(), 9);
		EXPECT_EQ(this->qr.x(), 39);
		EXPECT_EQ(this->qr.y(), 21);
		EXPECT_EQ(this->qr.z(), 57);
	}


	TYPED_TEST(QUAT_TEST_CLASS, ScalarDivisionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		typename TypeParam::Scalar scalar{1};

		// When
		auto result = this->q2 / scalar;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, QuaternionScalarDivision)
	{
		// When
		this->qr = this->q1 / 2;

		// Then
		EXPECT_EQ(this->qr.w(), 1.5);
		EXPECT_EQ(this->qr.x(), 6.5);
		EXPECT_EQ(this->qr.y(), 3.5);
		EXPECT_EQ(this->qr.z(), 9.5);
	}

#if !defined(MULTIPLICATION_UNSPECIALIZED)
	TYPED_TEST(QUAT_TEST_CLASS, MultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 * this->q2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}
#endif


	TYPED_TEST(QUAT_TEST_CLASS, I_x_I_eq_MinusOne)
	{
		// When
		this->qr = this->qi * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QUAT_TEST_CLASS, J_x_J_eq_MinusOne)
	{
		// When
		this->qr = this->qj * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QUAT_TEST_CLASS, K_x_K_eq_MinusOne)
	{
		// When
		this->qr = this->qk * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->q_one);
	}


	TYPED_TEST(QUAT_TEST_CLASS, I_x_J_eq_K)
	{
		// When
		this->qr = this->qi * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qk);
	}


	TYPED_TEST(QUAT_TEST_CLASS, J_x_K_eq_I)
	{
		// When
		this->qr = this->qj * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qi);
	}


	TYPED_TEST(QUAT_TEST_CLASS, K_x_I_eq_J)
	{
		// When
		this->qr = this->qk * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, this->qj);
	}


	TYPED_TEST(QUAT_TEST_CLASS, J_x_I_eq_MinusK)
	{
		// When
		this->qr = this->qj * this->qi;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qk);
	}


	TYPED_TEST(QUAT_TEST_CLASS, K_x_J_eq_MinusI)
	{
		// When
		this->qr = this->qk * this->qj;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qi);
	}


	TYPED_TEST(QUAT_TEST_CLASS, I_x_K_eq_MinusJ)
	{
		// When
		this->qr = this->qi * this->qk;

		// Then
		EXPECT_QUAT_EQ(this->qr, -this->qj);
	}


	TYPED_TEST(QUAT_TEST_CLASS, MultiplyLeftInverse_eq_1)
	{
		// When
		this->qr = Inverse(this->q1) * this->q1;

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}


	TYPED_TEST(QUAT_TEST_CLASS, MultiplyRightInverse_eq_1)
	{
		// When
		this->qr = this->q1 * Inverse(this->q1);

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}


	TYPED_TEST(QUAT_TEST_CLASS, DivisionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->q1 / this->q2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(QUAT_TEST_CLASS, Division)
	{
		// When
		this->qr = this->q1 / this->q1;

		// Then
		EXPECT_QUAT_NEAR(this->qr, this->q_one, 0.00001);
	}

}
