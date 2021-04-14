// Ark.cpp : Defines the entry point for the application.
//

#include "gtest/gtest.h"
#include "Ark/Math/Quaternion.h"

template<typename S>
using Quat = ark::math::Quat<S>;

TEST(QuatUnitTest, DefaultConstructor)
{
	Quat<int> q();
	SUCCEED();
}

TEST(QuatUnitTest, ElementConstructor)
{
	Quat<int> q(3, 5, 7, 11);
	EXPECT_EQ(q.w(), 3) << "The w element";
	EXPECT_EQ(q.x(), 5) << "The x element";
	EXPECT_EQ(q.y(), 7) << "The y element";
	EXPECT_EQ(q.z(), 11) << "The z element";
}

