/*************************************************************************
 * @file
 * @brief Vector Concept Unit Tests
 * 
 * @details Unit tests for the general, platform-independent vector 
 * expression tree implementation of vector functions.
 * 
 * @note There are no tests in this file related to constructors. 
 * Constructors are specific to concrete types. The abstract concept and 
 * expression trees do not address any aspect of vector construction 
 * beyond their own internal needs.
 * 
 * @author Noah Stein
 * @copyright © 2021 Noah Stein. All Rights Reserved.
 ************************************************************************/

//************************************************************************
//  Dependencies
//************************************************************************
#include "gtest/gtest.h"
#include "Ark/Math/Vector.h"

#include "Cfg.h"


//************************************************************************
//  Local Class Definitions
//************************************************************************
namespace ark::math::test::vector_unit_tests
{
	/*********************************************************************
	 * @brief Simple Concrete Vector Class
	 * 
	 * @tparam S The type of the scalar components
	 * 
	 * @tparam N The dimension of the vector
	 * 
	 * @details This is nearly the simplest implementation of a concrete 
	 * vector class for testing. As the general-purpose Vec class is much 
	 * more complex, it's ill-suited to test the Vector concept itself. 
	 * This class provides a simple implementation to avoid the tests 
	 * relying on external dependencies.
	 ********************************************************************/
	template<typename S, std::size_t N>
	class TestVec
	{
	public:
		using Scalar = S;

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

		template<typename ... T>
			requires (sizeof...(T) == N)
		constexpr TestVec(T && ... values) noexcept((std::is_nothrow_convertible_v<T, Scalar> && ...))
		{
			auto Set = [this, i = 0] <typename T> (T && value) mutable
			{
				data_[i++] = static_cast<Scalar>(std::forward<T>(value));
			};

			(Set(values), ...);
		}

		template<ark::math::Vector V>
		const TestVec& operator=(const V& rhs)
		{
			std::ranges::for_each(Range(), [&](std::size_t i) { data_[i] = static_cast<Scalar>(rhs(i)); });
			return *this;
		}
	};


	/*********************************************************************
	 * @brief Vector Unit Parametric Tests Fixture
	 * 
	 * @details The unit test fixture is designed with the focus around 
	 * testing 3-D vectors. There are some 2-D tests. There is a dearth 
	 * of 4-D tests as they have a low priority due to the presence of 
	 * tests for SIMD specializations.
	 ********************************************************************/
	template<typename C>
	class VectorUnitTest : public testing::Test
	{
	protected:
		/**
		 * @brief The scalar type of the vector to undergo testing
		 */
		using Scalar = typename C::Scalar;

		void SetUp() override
		{
			v1 = TestVec<Scalar, 3>{11, 13, 17};
			v2 = TestVec<Scalar, 3>{23, 29, 31};
		}

		TestVec<Scalar, 3> v1;
		TestVec<Scalar, 3> v2;
		TestVec<Scalar, 3> vr;
	};


	/**
	 * @brief Construct a new typed test suite object
	 */
	TYPED_TEST_SUITE(VectorUnitTest, StdTypes);


	//************************************************************************
	//  Tests
	//************************************************************************

	TYPED_TEST(VectorUnitTest, Negate)
	{
		// When
		this->vr = -this->v1;

		// Then
		EXPECT_EQ(this->vr(0), -11);
		EXPECT_EQ(this->vr(1), -13);
		EXPECT_EQ(this->vr(2), -17);
	}


	TYPED_TEST(VectorUnitTest, Add)
	{
		// When
		this->vr = this->v1 + this->v2;

		// Then
		EXPECT_EQ(this->vr(0), 34);
		EXPECT_EQ(this->vr(1), 42);
		EXPECT_EQ(this->vr(2), 48);
	}


	TYPED_TEST(VectorUnitTest, Subtract)
	{
		// When
		this->vr = this->v1 - this->v2;

		// Then
		EXPECT_EQ(this->vr(0), -12);
		EXPECT_EQ(this->vr(1), -16);
		EXPECT_EQ(this->vr(2), -14);
	}


	TYPED_TEST(VectorUnitTest, ScalarVectorMultiply)
	{
		// When
		this->vr = this->v1 * 10;

		// Then
		EXPECT_EQ(this->vr(0), 110);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 170);
	}


	TYPED_TEST(VectorUnitTest, VectorScalarMultiply)
	{
		// When
		this->vr = 10 * this->v1;

		// Then
		EXPECT_EQ(this->vr(0), 110);
		EXPECT_EQ(this->vr(1), 130);
		EXPECT_EQ(this->vr(2), 170);
	}


	TYPED_TEST(VectorUnitTest, VectorScalarDivide)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v = TestVec<Scalar, 3>{4, 10, 14};

		// When
		this->vr = v / 2;

		// Then
		EXPECT_EQ(this->vr(0), 2);
		EXPECT_EQ(this->vr(1), 5);
		EXPECT_EQ(this->vr(2), 7);
	}


	TYPED_TEST(VectorUnitTest, EqualityCheckSame)
	{
		// When
		bool result = this->v1 == this->v1;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(VectorUnitTest, EqualityCheckDifferent)
	{
		// When
		bool result = this->v1 == this->v2;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(VectorUnitTest, InequalityCheckSame)
	{
		// When
		bool result = this->v1 != this->v1;

		// Then
		EXPECT_FALSE(result);
	}


	TYPED_TEST(VectorUnitTest, InequalityCheckDifferent)
	{
		// When
		bool result = this->v1 != this->v2;

		// Then
		EXPECT_TRUE(result);
	}


	TYPED_TEST(VectorUnitTest, VectorDotProduct)
	{
		// When
		auto result = Dot(this->v1, this->v2);

		// Then
		EXPECT_EQ(result, 1157);
	}


	TYPED_TEST(VectorUnitTest, VectorCrossProduct2D)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v1 = TestVec<Scalar, 2>{5, 3};
		auto v2 = TestVec<Scalar, 2>{2, 7};

		// When
		auto result = Cross(v1, v2);

		// Then
		EXPECT_EQ(result, 29);
	}


	TYPED_TEST(VectorUnitTest, VectorCrossProduct3D)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v1 = TestVec<Scalar, 3>{2, 3, 5};
		auto v2 = TestVec<Scalar, 3>{7, 11, 13};

		// When
		this->vr = Cross(v1, v2);

		// Then
		EXPECT_EQ(this->vr(0), -16);
		EXPECT_EQ(this->vr(1), 9);
		EXPECT_EQ(this->vr(2), 1);
	}


	TYPED_TEST(VectorUnitTest, VectorCrossProduct4D)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v1 = TestVec<Scalar, 4>{2, 3, 5, 7};
		auto v2 = TestVec<Scalar, 4>{7, 11, 13, 17};

		// When
		auto result = Cross(v1, v2);

		// Then
		EXPECT_EQ(result(0), -16);
		EXPECT_EQ(result(1), 9);
		EXPECT_EQ(result(2), 1);
		EXPECT_EQ(result(3), 0);
	}


	TYPED_TEST(VectorUnitTest, VectorNorm)
	{
		// Given
		using Scalar = typename TypeParam::Scalar;
		auto v = TestVec<Scalar, 4>{2, 4, 2, 1};

		// When
		auto result = Norm(v);

		// Then
		EXPECT_EQ(result, 5);
	}
}
