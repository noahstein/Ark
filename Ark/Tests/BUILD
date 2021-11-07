#	Copyright (c) 2021 Noah Stein. All Rights Reserved.

cc_test(
	name = "Tests",
	size = "small",
	srcs = [
		"MatrixUnitTests.cpp",
		"MtxUnitTests.cpp",
		"QuaternionUnitTests.cpp",
		"QuatUnitTests.cpp",
		"VectorUnitTests.cpp",
		"VecUnitTests.cpp",
		"Vec4UnitTests.cpp",

		"Cfg.h",
	],
	deps = [
		"@gtest//:gtest",
		"@gtest//:gtest_main",
		"//Ark:Ark",
	],
	defines = [
		"HAL_SIMD=Sse2"
	]
)