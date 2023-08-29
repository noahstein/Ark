/*************************************************************************
 * @file
 * @brief Parameterized 4-D Vec Unit Tests
 *
 * @details Unit tests for Vec classes specifically designed to test 4-D
 * functionality. This is desired for a few reasons:
 *
 * -# 3-D games heavily rely upon 4-D vectors because they are a 
 * homogeneous representation of 3-D vectors and points.
 * 
 * -# SIMD architectures tend to be optimized around 4-element data, so 
 * it's usually more efficient to use 4-D vectors.
 *
 * -# There are some 3-D/4-D specific functions. E..g. the 2-D cross 
 * product is different than that used in 3-D. The 2-D version returns a 
 * scalar whereas the 3-D version returns a vector.
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Vec.h"


//************************************************************************
//  Test Suite
//************************************************************************

namespace ark::math::test::VEC4_TEST_NAMESPACE
{
	using namespace ::ark::hal;
	using namespace ::ark::math;


	/*********************************************************************
	 * @brief Parametric 2-D Vec Unit Tests Fixture
	 *
	 * @tparam C A configuration class that specifies the type of the
	 * scalar and the SIMD ISA tag.
	 *
	 * @details The parametric fixture has two main design features:
	 *
	 * -# It is designed as a template Google Test template class, so it
	 * can be used to test both single- and double-precision floating
	 * point vectors.
	 *
	 * -# It will not work properly stand-alone. It must be included
	 * in .cpp files that configure the tests via macros and a type alias.
	 * An example ks referenced in the See Also section.
	 *
	 * @sa Cfg.h
	 * @sa Vec4Sse3UnitTests.cpp
	 ********************************************************************/
	template<typename C>
	class VEC4_TEST_CLASS : public testing::Test
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

		/// @name Support Functions
		/// @{

		/*****************************************************************
		 * @brief Is a result of a basic expression type?
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if basic expression, false if optimized type
		 *
		 * @details THis predicate checks to see if the type of its
		 * parameter is of a basic expression template type. This is
		 * useful in determining if an optimized function is running
		 * instead of the slower, generic implementation.
		 */
		template<typename R>
		bool IsResultTypeBasicExpression(R& result) const
		{
			bool derived = std::derived_from<R, VectorExpr>;
			bool same = std::same_as<VecBasic<Scalar, 4>, Vec<Scalar, 4, Isa>>;
			return derived && same;
		}


		/*****************************************************************
		 * @brief Is the result type the same as that being tested?
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if types are same, false if different
		 *
		 * @details Predicate determines if the type of the test is the
		 * same as that being tested. This is useful to test if optimized
		 * functions are getting called as they return the same type
		 * that's being tested whereas the generic functions return some
		 * sort of expression template type.
		 */
		template<typename R>
		bool IsResultTypeSame(R& result) const
		{
			return std::same_as<R, Vec<Scalar, 4, Isa>>;
		}


		/*****************************************************************
		 * @brief Determine if a function is properly specialized
		 *
		 * @tparam R The type of the quaternion result
		 * @param result A result from a quaternion expression under test
		 * @result True if types are properly optimized, false if not.
		 *
		 * @details A function is properly specialized in two cases:
		 *
		 * -# The input vectors are generic vectors and the output vector
		 * is some form of expression template.
		 *
		 * -# The input vectors are a SIMD-optimized class and the output
		 * vector is that same optimized class. If it's an expression
		 * template then the generic function executed, not one
		 * specifically implemented for the SIMD ISA.
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


	/*****************************************************************
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(VEC4_TEST_CLASS, TestTypes);


	//************************************************************************
	// Tests
	//************************************************************************

	//========================================================================
	// Constructors and Fundamental Member Functions
	//========================================================================

	TYPED_TEST(VEC4_TEST_CLASS, DefaultConstructor)
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


	TYPED_TEST(VEC4_TEST_CLASS, ComponentConstructor)
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


	TYPED_TEST(VEC4_TEST_CLASS, CopyConstructor)
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


	TYPED_TEST(VEC4_TEST_CLASS, OperatorEqual)
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


	TYPED_TEST(VEC4_TEST_CLASS, AccessorAt0)
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


	TYPED_TEST(VEC4_TEST_CLASS, AccessorAt1)
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


	TYPED_TEST(VEC4_TEST_CLASS, AccessorAt2)
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


	TYPED_TEST(VEC4_TEST_CLASS, AccessorAt3)
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

	TYPED_TEST(VEC4_TEST_CLASS, NegationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = -this->v1;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, Negate)
	{
		// When
		this->vr = -this->v1;

		// Then
		EXPECT_EQ(this->vr(0), -3);
		EXPECT_EQ(this->vr(1), -13);
		EXPECT_EQ(this->vr(2), -7);
		EXPECT_EQ(this->vr(3), -19);
	}


	TYPED_TEST(VEC4_TEST_CLASS, AdditionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->v1 + this->v2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, Add)
	{
		// When
		this->vr = this->v1 + this->v2;

		// Then
		EXPECT_EQ(this->vr(0), 8);
		EXPECT_EQ(this->vr(1), 24);
		EXPECT_EQ(this->vr(2), 30);
		EXPECT_EQ(this->vr(3), 48);
	}


	TYPED_TEST(VEC4_TEST_CLASS, SubtractionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// When
		auto result = this->v1 - this->v2;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, Subtract)
	{
		// When
		this->vr = this->v1 - this->v2;

		// Then
		EXPECT_EQ(this->vr(0), -2);
		EXPECT_EQ(this->vr(1), 2);
		EXPECT_EQ(this->vr(2), -16);
		EXPECT_EQ(this->vr(3), -10);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorScalarMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		auto result = this->v1 * Scalar{2.0f};

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorScalarMultiply)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		this->vr = this->v1 * Scalar{10};

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 70);
		EXPECT_EQ(this->vr(3), 190);
	}


	TYPED_TEST(VEC4_TEST_CLASS, ScalarVectorMultiplicationReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		auto result = Scalar{2.0f} *this->v1;

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, ScalarVectorMultiply)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		this->vr = Scalar{10} *this->v1;

		// Then
		EXPECT_EQ(this->vr(0), 30);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 70);
		EXPECT_EQ(this->vr(3), 190);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorScalarDivisionReturnsSpecializedTypeOrExpressionCorrectly)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;

		// When
		auto result =  this->v1 / Scalar{2.0f};

		// Then
		bool correct = this->IsSpecializedCorrectly(result);
		EXPECT_TRUE(correct);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorScalarDivide)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;
		auto v = Vec<Scalar, 4, Isa>{4, 6, 10, 14};

		// When
		this->vr = v / Scalar{2};

		// Then
		EXPECT_EQ(this->vr(0), 2);
		EXPECT_EQ(this->vr(1), 3);
		EXPECT_EQ(this->vr(2), 5);
		EXPECT_EQ(this->vr(3), 7);
	}


	TYPED_TEST(VEC4_TEST_CLASS, EqualityCheckSame)
	{
		// When
		bool result = this->v1 == this->v1;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(VEC4_TEST_CLASS, EqualityCheckDifferent)
	{
		// When
		bool result = this->v1 == this->v2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(VEC4_TEST_CLASS, InequalityCheckSame)
	{
		// When
		bool result = this->v1 != this->v1;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(VEC4_TEST_CLASS, InequalityCheckDifferent)
	{
		// When
		bool result = this->v1 != this->v2;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorDotProduct)
	{
		// When
		float result = Dot(this->v1, this->v2);

		// Then
		EXPECT_EQ(result, 870);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorCrossProduct4D)
	{
		// When
		this->vr = Cross(this->v1, this->v2);

		// Then
		EXPECT_EQ(this->vr(0), 222);
		EXPECT_EQ(this->vr(1), -34);
		EXPECT_EQ(this->vr(2), -32);
		EXPECT_EQ(this->vr(3), 0);
	}


	TYPED_TEST(VEC4_TEST_CLASS, VectorNorm)
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
