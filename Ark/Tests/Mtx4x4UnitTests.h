/*************************************************************************
 * @file
 * @brief Parameterized 4x4 Mtx Unit Tests
 *
 * @details Unit tests for Mtx classes specifically designed to test the
 * functionality of 4x4 matrices. This is important because 3D graphics 
 * relies heavily upon matrices to represent transformations and 4-D 
 * matrices can perform linear transformation and perspective projection 
 * of homogeneous 3-D points and vectors.
 *
 * @author Noah Stein
 * @copyright © 2023 Noah Stein. All Rights Reserved.
 ************************************************************************/

 //************************************************************************
 //  Dependencies
 //************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Mtx.h"


//************************************************************************
//  Test Suite
//************************************************************************

/*************************************************************************
 * @brief Macro to test if a 4x4 matrix's elements are equal to a 
 * collection of known constants.
 */
#define EXPECT_MTX4x4_EQ(m,        \
		m00, m01, m02, m03,        \
		m10, m11, m12, m13,        \
		m20, m21, m22, m23,        \
		m30, m31, m32, m33)        \
		                           \
		EXPECT_EQ((m)(0, 0), m00); \
		EXPECT_EQ((m)(0, 1), m01); \
		EXPECT_EQ((m)(0, 2), m02); \
		EXPECT_EQ((m)(0, 3), m03); \
		                           \
		EXPECT_EQ((m)(1, 0), m10); \
		EXPECT_EQ((m)(1, 1), m11); \
		EXPECT_EQ((m)(1, 2), m12); \
		EXPECT_EQ((m)(1, 3), m13); \
		                           \
		EXPECT_EQ((m)(2, 0), m20); \
		EXPECT_EQ((m)(2, 1), m21); \
		EXPECT_EQ((m)(2, 2), m22); \
		EXPECT_EQ((m)(2, 3), m23); \
		                           \
		EXPECT_EQ((m)(3, 0), m30); \
		EXPECT_EQ((m)(3, 1), m31); \
		EXPECT_EQ((m)(3, 2), m32); \
		EXPECT_EQ((m)(3, 3), m33)


namespace ark::math::test::MTX4X4_TEST_NAMESPACE
{
	/*********************************************************************
	 * @brief Parametric 4x4 Mtx Unit Tests Fixture
	 *
	 * @tparam C A configuration class that specifies the type of the
	 * scalar and the SIMD ISA tag.
	 *
	 * @details The parametric fixture has two main design features:
	 *
	 * -# It is designed as a template Google Test template class, so it
	 * can be used to test both single- and double-precision floating
	 * point 4x4 matrices.
	 *
	 * -# It will not work properly stand-alone. It must be included
	 * in .cpp files that configure the tests via macros and a type alias.
	 * An example ks referenced in the See Also section.
	 *
	 * @sa Cfg.h
	 * @sa Mtx4x4UnitTests.cpp
	 ********************************************************************/
	template<typename C>
	class MTX4X4_TEST_CLASS : public testing::Test
	{
	protected:
		using Scalar = typename C::Scalar;
		using Isa = typename C::Isa;

		Mtx<Scalar, 4, 4 > m1;
		Mtx<Scalar, 4, 4 > m2;
		Mtx<Scalar, 4, 4 > mr;

		/// @brief Setup the standard 4x4 matrices
		void SetUp() override
		{
			m1 = Mtx<Scalar, 4, 4 >
			{
				 1,  3,  5,  7,
				 9, 11, 13, 15,
				17, 19, 21, 23,
				25, 27, 29, 31
			};

			m2 = Mtx<Scalar, 4, 4 >
			{
				 2,  4,  6,  8,
				10, 12, 14, 16,
				18, 20, 22, 24,
				26, 28, 30, 32
			};
		}
	};


	/*****************************************************************
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(MTX4X4_TEST_CLASS, TestTypes);


	//************************************************************************
	// Tests
	//************************************************************************

	//========================================================================
	// Constructors
	//========================================================================

	TYPED_TEST(MTX4X4_TEST_CLASS, DefaultConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		Mtx<Scalar, 4, 4, Isa> v;

		// Then
		SUCCEED();
	}

	TYPED_TEST(MTX4X4_TEST_CLASS, ElementConstructor)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		// When
		auto m = Mtx<Scalar, 4, 4, Isa>
		{
			 1,  3,  5,  7,
			 9, 11, 13, 15,
			17, 19, 21, 23,
			25, 27, 29, 31
		};

		// Then
		EXPECT_MTX4x4_EQ
		(	m,
			 1,  3,  5,  7,
			 9, 11, 13, 15,
			17, 19, 21, 23,
			25, 27, 29, 31
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, MatrixCopyConstructor)
	{
		// When
		auto m{this->m2};

		// Then
		EXPECT_MTX4x4_EQ
		(	m,
			 2,  4,  6 , 8,
			10, 12, 14, 16,
			18, 20, 22, 24,
			26, 28, 30, 32
		);
	}


	//========================================================================
	// Dimension Functions
	//========================================================================

	TYPED_TEST(MTX4X4_TEST_CLASS, Width)
	{
		// When
		std::size_t width = this->m1.Width();

		// Then
		EXPECT_EQ(width, 4);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Height)
	{
		// When
		std::size_t height = this->m1.Height();

		// Then
		EXPECT_EQ(height, 4);
	}


	//========================================================================
	// Accessor
	//========================================================================

	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_0_0)
	{
		// When
		auto v = this->m1(0, 0);

		// Then
		EXPECT_EQ(v, 1);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_0_1)
	{
		// When
		auto v = this->m1(0, 1);

		// Then
		EXPECT_EQ(v, 3);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_0_2)
	{
		// When
		auto v = this->m1(0, 2);

		// Then
		EXPECT_EQ(v, 5);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_0_3)
	{
		// When
		auto v = this->m1(0, 3);

		// Then
		EXPECT_EQ(v, 7);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_1_0)
	{
		// When
		auto v = this->m1(1, 0);

		// Then
		EXPECT_EQ(v, 9);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_1_1)
	{
		// When
		auto v = this->m1(1, 1);

		// Then
		EXPECT_EQ(v, 11);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_1_2)
	{
		// When
		auto v = this->m1(1, 2);

		// Then
		EXPECT_EQ(v, 13);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_1_3)
	{
		// When
		auto v = this->m1(1, 3);

		// Then
		EXPECT_EQ(v, 15);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_2_0)
	{
		// When
		auto v = this->m1(2, 0);

		// Then
		EXPECT_EQ(v, 17);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_2_1)
	{
		// When
		auto v = this->m1(2, 1);

		// Then
		EXPECT_EQ(v, 19);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_2_2)
	{
		// When
		auto v = this->m1(2, 2);

		// Then
		EXPECT_EQ(v, 21);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_2_3)
	{
		// When
		auto v = this->m1(2, 3);

		// Then
		EXPECT_EQ(v, 23);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_3_0)
	{
		// When
		auto v = this->m1(3, 0);

		// Then
		EXPECT_EQ(v, 25);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_3_1)
	{
		// When
		auto v = this->m1(3, 1);

		// Then
		EXPECT_EQ(v, 27);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_3_2)
	{
		// When
		auto v = this->m1(3, 2);

		// Then
		EXPECT_EQ(v, 29);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Accessor_3_3)
	{
		// When
		auto v = this->m1(3, 3);

		// Then
		EXPECT_EQ(v, 31);
	}


	//========================================================================
	// Mathematical Functions
	//========================================================================

	TYPED_TEST(MTX4X4_TEST_CLASS, Negate)
	{
		// When
		this->mr = -this->m1;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			 -1,  -3,  -5,  -7,
			 -9, -11, -13, -15,
			-17, -19, -21, -23,
			-25, -27, -29, -31
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Add)
	{
		// When
		this->mr = this->m1 + this->m2;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			3,   7, 11, 15,
			19, 23, 27, 31,
			35, 39, 43, 47,
			51, 55, 59, 63
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Subtract)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		auto m = Mtx<Scalar, 4, 4, Isa>
		{
			55, 84, 77, 60,
			96, 76, 81, 63,
			65, 92, 71, 57,
			89, 78, 90, 66
		};

		// When
		this->mr = m - this->m1;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			54, 81, 72, 53,
			87, 65, 68, 48,
			48, 73, 50, 34,
			64, 51, 61, 35
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, ScalarMatrixMultiplication)
	{
		// When
		this->mr = 10.0f * this->m1;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			 10,  30,  50,  70,
			 90, 110, 130, 150,
			170, 190, 210, 230,
			250, 270, 290, 310
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, MatrixScalarMultiplication)
	{
		// When
		this->mr = 10.0f * this->m1;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			 10,  30,  50,  70,
			 90, 110, 130, 150,
			170, 190, 210, 230,
			250, 270, 290, 310
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, MatrixScalarDivision)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		auto m = Mtx<Scalar, 4, 4, Isa>
		{
			36, 84, 72, 60,
			96, 76, 82, 62,
			66, 92, 70, 58,
			84, 78, 90, 66
		};

		// When
		this->mr = m / 2.0f;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			18, 42, 36, 30,
			48, 38, 41, 31,
			33, 46, 35, 29,
			42, 39, 45, 33
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Transpose)
	{
		// When
		this->mr = Trans(this->m1);

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			 1,  9, 17, 25,
			 3, 11, 19, 27,
			 5, 13, 21, 29,
			 7, 15, 23, 31
		);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Determinant)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		using Isa = typename TypeParam::Isa;

		auto m = Mtx<Scalar, 4, 4, Isa>
		{
			9, 4, 7, 3,
			8, 5, 6, 4,
			2, 2, 5, 6,
			3, 7, 8, 4
		};
		// When
		auto determinant = Det(m);

		// Then
		EXPECT_EQ(determinant, -471);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, Multiplication)
	{
		// When
		this->mr = this->m1 * this->m2;

		// Then
		EXPECT_MTX4x4_EQ
		(	this->mr,
			 304,  336,  368,  400,
			 752,  848,  944, 1040,
			1200, 1360, 1520, 1680,
			1648, 1872, 2096, 2320
		);
	}


	//========================================================================
	// Comparisons
	//========================================================================

	TYPED_TEST(MTX4X4_TEST_CLASS, EqualityCheckSame)
	{
		// When
		bool result = this->m1 == this->m1;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, EqualityCheckDifferent)
	{
		// When
		bool result = this->m1 == this->m2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, InequalityCheckSame)
	{
		// When
		bool result = this->m2 != this->m2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(MTX4X4_TEST_CLASS, InequalityCheckDifferent)
	{
		// When
		bool result = this->m1 != this->m2;

		// Then
		EXPECT_TRUE(result);
	}
}
