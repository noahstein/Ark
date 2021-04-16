#	Copyright (c) 2021 Noah Stein. All Rights Reserverd.

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
		"//Ark:Ark",
	],
	defines = [
		"HAL_SIMD=Sse"
	]
)