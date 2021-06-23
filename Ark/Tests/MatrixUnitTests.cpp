/*========================================================================
Description
	Unit tests for functions related to the ark::math::Matrix concept, 
	and all basic functionality implemented against that concept.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/


/*========================================================================
 Dependencies
========================================================================*/
#include "gtest/gtest.h"
#include "Ark/Math/Matrix.h"


/*========================================================================
 Local Class Definitions
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
		TestMtx class implements the Matrix concept in a simple way to 
		test the functionality implemented against the concept
	--------------------------------------------------------------------*/
	template<int R, int C>
	class TestMtx
	{
	public:
		using Scalar = float;

	private:
		Scalar data_[R][C];

	public:
		static constexpr std::size_t Width() { return C; }
		static constexpr std::size_t Height() { return R; }
		constexpr Scalar operator()(std::size_t row, std::size_t column) const
		{
			  return data_[row][column];
		}

		TestMtx() = default;

		template<Matrix M>
		constexpr TestMtx(const M& rhs)
		{
			for (std::size_t r = 0; r < rhs.Height(); ++r)
			{
				for (std::size_t c = 0; c < rhs.Width(); ++c)
				{
					data_[r][c] = rhs(r, c);					
				}
			}
		}

		template<typename ... T>
			requires (sizeof...(T) == R * C)
		constexpr TestMtx(T && ... values)
		{
			auto Set = [this, r = 0, c = 0] <typename T> (T && value) mutable
			{
				data_[r][c] = static_cast<Scalar>(std::forward<T>(value));
				if (++c == Width())
				{
					c = 0;
					++r;
				}
			};

			(Set(values), ...);
		}

		template<ark::math::Matrix M>
		constexpr TestMtx& operator=(const M& rhs)
		{
			for (std::size_t r = 0; r < rhs.Height(); ++r)
			{
				for (std::size_t c = 0; c < rhs.Width(); ++c)
				{
					data_[r][c] = rhs(r, c);					
				}
			}
			return *this;
		}
	};
}
using Mtx = ark::math::TestMtx<2, 2>;


/*------------------------------------------------------------------------
	Vector Test Fixture
------------------------------------------------------------------------*/
class MatrixUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		m = Mtx({ 2.0f, 3.0f,
		          5.0f, 7.0f});
		m1 = Mtx({ 2.0f, 3.0f,
		           5.0f, 7.0f});
		m2 = Mtx({ 11.0f, 13.0f,
		           17.0f, 19.0f});
	}

	Mtx m;
	Mtx m1;
	Mtx m2;
	Mtx mr;
};


/*========================================================================
 Tests
========================================================================*/

TEST_F(MatrixUnitTests, Negate)
{
	// When
	mr = -m;

	// Then
	EXPECT_EQ(mr(0, 0), -2.0f);
	EXPECT_EQ(mr(0, 1), -3.0f);
	EXPECT_EQ(mr(1, 0), -5.0f);
	EXPECT_EQ(mr(1,1 ), -7.0f);
}


TEST_F(MatrixUnitTests, Add)
{
	// When
	mr = m1 + m2;

	// Then
	EXPECT_EQ(mr(0, 0), 13.0f);
	EXPECT_EQ(mr(0, 1), 16.0f);
	EXPECT_EQ(mr(1, 0), 22.0f);
	EXPECT_EQ(mr(1,1 ), 26.0f);
}


TEST_F(MatrixUnitTests, Subtract)
{
	// When
	mr = m1 - m2;

	// Then
	EXPECT_EQ(mr(0, 0), -9.0f);
	EXPECT_EQ(mr(0, 1), -10.0f);
	EXPECT_EQ(mr(1, 0), -12.0f);
	EXPECT_EQ(mr(1,1 ), -12.0f);
}


TEST_F(MatrixUnitTests, ScalarMatrixMultiplication)
{
	// When
	mr = 10.0f * m;

	// Then
	EXPECT_EQ(mr(0, 0), 20.0f);
	EXPECT_EQ(mr(0, 1), 30.0f);
	EXPECT_EQ(mr(1, 0), 50.0f);
	EXPECT_EQ(mr(1,1 ), 70.0f);
}


TEST_F(MatrixUnitTests, MatrixScalarMultiplication)
{
	// When
	mr = m * 10.0f;

	// Then
	EXPECT_EQ(mr(0, 0), 20.0f);
	EXPECT_EQ(mr(0, 1), 30.0f);
	EXPECT_EQ(mr(1, 0), 50.0f);
	EXPECT_EQ(mr(1,1 ), 70.0f);
}


TEST_F(MatrixUnitTests, MatrixScalarDivision)
{
	// When
	mr = m / 2.0f;

	// Then
	EXPECT_EQ(mr(0, 0), 1.0f);
	EXPECT_EQ(mr(0, 1), 1.5f);
	EXPECT_EQ(mr(1, 0), 2.5f);
	EXPECT_EQ(mr(1,1 ), 3.5f);
}


TEST_F(MatrixUnitTests, EqualityCheckSame)
{
	// When
	bool result = m == m;

	// Then
	EXPECT_TRUE(result);
}


TEST_F(MatrixUnitTests, EqualityCheckDifferent)
{
	// When
	bool result = m1 == m2;

	// Then
	EXPECT_FALSE(result);
}


TEST_F(MatrixUnitTests, InequalityCheckSame)
{
	// When
	bool result = m != m;

	// Then
	EXPECT_FALSE(result);
}


TEST_F(MatrixUnitTests, InequalityCheckDifferent)
{
	// When
	bool result = m1 != m2;

	// Then
	EXPECT_TRUE(result);
}


TEST_F(MatrixUnitTests, Multiplication)
{
	// When
	mr = m1 * m2;

	// Then
	EXPECT_EQ(mr(0, 0), 73.0f);
	EXPECT_EQ(mr(0, 1), 83.0f);
	EXPECT_EQ(mr(1, 0), 174.0f);
	EXPECT_EQ(mr(1,1 ), 198.0f);
}


TEST_F(MatrixUnitTests, Determinant2x2)
{
	// When
	float result = Det(m);

	// Then
	EXPECT_EQ(result, -1.0f);
}


TEST_F(MatrixUnitTests, Determinant3x3)
{
	// Given
	ark::math::TestMtx<3, 3> mtx({5, 2, 3,
	                              4, 5, 6,
	                              7, 8, 9});
	// When
	float result = Det(mtx);

	// Then
	EXPECT_EQ(result, -12.0f);
}


TEST_F(MatrixUnitTests, Determinant4x45)
{
	// Given
	ark::math::TestMtx<4, 4> mtx({1, 3, 5, 9,
	                              1, 3, 1, 7,
	                              4, 3, 9, 7,
	                              5, 2, 0, 9});
	// When
	float result = Det(mtx);

	// Then
	EXPECT_EQ(result, -376);
}
