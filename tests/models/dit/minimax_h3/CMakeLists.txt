include(cc_test)

cc_test(
  NAME
    minimax_h3_test
  SRCS
    minimax_h3_test.cpp
  DEPS
    :model_loader
    state_dict
    torch
    GTest::gtest_main
  COPTS
    -DXLLM_MINIMAX_H3_DISABLE_REGISTRATION
)
