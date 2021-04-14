"""Ark library test executable"""

cc_test(
	name = "ArkTests",
	size = "small",
	srcs = [
		"QuaternionUnitTests.cpp",
		"QuatUnitTests.cpp",
	],
	deps = [
		"@gtest//:gtest",
		"@gtest//:gtest_main",
		"//Ark/Math:Ark-Math",
	]
)