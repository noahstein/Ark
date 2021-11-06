/*========================================================================
Description
	Unit tests for functions related to the ark::math::Quaternion concept,
	and all basic functionality implemented against that concept.

Copyright
	Copyright (c) 2021 Noah Stein. All Rights Reserved.
========================================================================*/


/*========================================================================
 Dependencies
========================================================================*/
#include "gtest/gtest.h"
#include "Ark/Math/Vector.h"


/*========================================================================
 Local Class Definitions
========================================================================*/
namespace ark::math
{
	/*--------------------------------------------------------------------
		TestVec class implements the Vector concept in a simple way to
		test the functionality implemented against the concept
	--------------------------------------------------------------------*/
	template<std::size_t N>
	class TestVec
	{
	public:
		using Scalar = float;

	private:
		Scalar data_[N];

		static constexpr auto Range()
		{
			return std::views::iota(std::size_t{ 0 }, N);
		}

	public:
		static constexpr std::size_t Size() { return N; } const
		constexpr Scalar operator()(std::size_t index) const { return data_[index]; }

		TestVec() = default;

		template<ark::math::Vector V>
		TestVec(const V& rhs)
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(rhs(i)); });
		}

		constexpr TestVec(const Scalar(&values)[N])
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(values[i]); });
		}

		template<ark::math::Vector V>
		const TestVec& operator=(const V& rhs)
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(rhs(i)); });
			return *this;
		}
	};
}
using Vec = ark::math::TestVec<4>;


/*------------------------------------------------------------------------
	Vector Test Fixture
------------------------------------------------------------------------*/
class VectorUnitTests : public testing::Test
{
protected:
	void SetUp() override
	{
		v = Vec({2.0f, 3.0f, 5.0f, 7.0f});
		v1 = Vec({11.0f, 13.0f, 17.0f, 19.0f});
		v2 = Vec({23.0f, 29.0f, 31.0f, 37.0f});
	}

	Vec v;
	Vec v1;
	Vec v2;
	Vec vr;
};


/*========================================================================
 Tests
========================================================================*/

TEST_F(VectorUnitTests, Negate)
{
	// When
	vr = -v;

	// Then
	EXPECT_EQ(vr(0), -2.0f);
	EXPECT_EQ(vr(1), -3.0f);
	EXPECT_EQ(vr(2), -5.0f);
	EXPECT_EQ(vr(3), -7.0f);
}

TEST_F(VectorUnitTests, Add)
{
	// When
	vr = v1 + v2;

	// Then
	EXPECT_EQ(vr(0), 34.0f);
	EXPECT_EQ(vr(1), 42.0f);
	EXPECT_EQ(vr(2), 48.0f);
	EXPECT_EQ(vr(3), 56.0f);
}

TEST_F(VectorUnitTests, Subtract)
{
	// When
	vr = v1 - v2;

	// Then
	EXPECT_EQ(vr(0), -12.0f);
	EXPECT_EQ(vr(1), -16.0f);
	EXPECT_EQ(vr(2), -14.0f);
	EXPECT_EQ(vr(3), -18.0f);
}

TEST_F(VectorUnitTests, ScalarVectorMultiply)
{
	// When
	vr = v * 10.0f;

	// Then
	EXPECT_EQ(vr(0), 20.0f);
	EXPECT_EQ(vr(1), 30.0f);
	EXPECT_EQ(vr(2), 50.0f);
	EXPECT_EQ(vr(3), 70.0f);
}

TEST_F(VectorUnitTests, VectorScalarMultiply)
{
	// When
	vr = 10.0f * v;

	// Then
	EXPECT_EQ(vr(0), 20.0f);
	EXPECT_EQ(vr(1), 30.0f);
	EXPECT_EQ(vr(2), 50.0f);
	EXPECT_EQ(vr(3), 70.0f);
}

TEST_F(VectorUnitTests, VectorScalarDivide)
{
	// When
	vr = v / 2.0f;

	// Then
	EXPECT_EQ(vr(0), 1.0f);
	EXPECT_EQ(vr(1), 1.5f);
	EXPECT_EQ(vr(2), 2.5f);
	EXPECT_EQ(vr(3), 3.5f);
}

TEST_F(VectorUnitTests, EqualityCheckSame)
{
	// When
	bool result = v1 == v1;

	// Then
	EXPECT_TRUE(result);
}

TEST_F(VectorUnitTests, EqualityCheckDifferent)
{
	// When
	bool result = v1 == v2;

	// Then
	EXPECT_FALSE(result);
}

TEST_F(VectorUnitTests, InequalityCheckSame)
{
	// When
	bool result = v1 != v1;

	// Then
	EXPECT_FALSE(result);
}

TEST_F(VectorUnitTests, InequalityCheckDifferent)
{
	// When
	bool result = v1 != v2;

	// Then
	EXPECT_TRUE(result);
}

TEST_F(VectorUnitTests, VectorDotProduct)
{
	// When
	float result = Dot(v1, v2);

	// Then
	EXPECT_EQ(result, 1860);
}

TEST_F(VectorUnitTests, VectorCrossProduct2D)
{
	// Given
	ark::math::TestVec<2> v1({5.0f, 3.0f});
	ark::math::TestVec<2> v2({2.0f, 7.0f});

	// When
	float result = Cross(v1, v2);

	// Then
	EXPECT_EQ(result, 29.0f);
}

TEST_F(VectorUnitTests, VectorCrossProduct3D)
{
	// Given
	ark::math::TestVec<3> v1({2.0f, 3.0f, 5.0f});
	ark::math::TestVec<3> v2({7.0f, 11.0f, 13.0f});

	// When
	ark::math::TestVec<3> result = Cross(v1, v2);

	// Then
	EXPECT_EQ(result(0), -16.0f);
	EXPECT_EQ(result(1), 9.0f);
	EXPECT_EQ(result(2), 1.0f);
}

TEST_F(VectorUnitTests, VectorCrossProduct4D)
{
	// Given
	ark::math::TestVec<4> v1({2.0f, 3.0f, 5.0f, 7.0f});
	ark::math::TestVec<4> v2({7.0f, 11.0f, 13.0f, 17.0f});

	// When
	ark::math::TestVec<4> result = Cross(v1, v2);

	// Then
	EXPECT_EQ(result(0), -16.0f);
	EXPECT_EQ(result(1), 9.0f);
	EXPECT_EQ(result(2), 1.0f);
}

TEST_F(VectorUnitTests, VectorNorm)
{
	// When
	float result = Norm(v);

	// Then
	EXPECT_EQ(result, std::sqrt(87.0f));
}
