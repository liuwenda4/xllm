include(cc_test)

cc_test(
  NAME
    mm_codec_test
  SRCS
    mm_codec_test.cpp
  DEPS
    :multimodal
    GTest::gtest_main
)
target_link_libraries(mm_codec_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
