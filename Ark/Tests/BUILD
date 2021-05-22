#	Copyright (c) 2021 Noah Stein. All Rights Reserverd.

cc_test(
	name = "ArkTests",
	size = "small",
	srcs = [
		"MatrixUnitTests.cpp",
		"MtxUnitTests.cpp",
		"QuaternionUnitTests.cpp",
		"QuatUnitTests.cpp",
		"VectorUnitTests.cpp",
		"VecUnitTests.cpp",
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