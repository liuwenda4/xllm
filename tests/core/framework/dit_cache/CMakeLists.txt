include(cc_test)

cc_test(
  NAME
    similarity_residual_cache_test
  SRCS
    similarity_residual_cache_test.cpp
  DEPS
    dit_cache
    GTest::gtest_main
)
