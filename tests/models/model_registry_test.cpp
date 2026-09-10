/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "models/model_registry.h"

#include <gtest/gtest.h>

#include <string>

namespace xllm {
namespace {

TEST(ModelRegistryTest, DeepseekV4DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v4_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v4_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, MiniMaxH3ResolvesExactPipelineAndConfigClassNames) {
  EXPECT_TRUE(ModelRegistry::has_dit_model_factory("MiniMaxH3Pipeline"));
  EXPECT_TRUE(static_cast<bool>(
      ModelRegistry::get_model_args_loader("MiniMaxH3DiTModel")));
  EXPECT_TRUE(static_cast<bool>(
      ModelRegistry::get_model_args_loader("MiniMaxH3Transformer3DModel")));
  EXPECT_TRUE(static_cast<bool>(
      ModelRegistry::get_model_args_loader("MiniMaxH3VideoVAE")));
  EXPECT_TRUE(static_cast<bool>(
      ModelRegistry::get_model_args_loader("MiniMaxH3AudioVAE")));
  EXPECT_TRUE(static_cast<bool>(
      ModelRegistry::get_model_args_loader("MiniMaxH3Qwen3VLHFEncoder")));

  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  EXPECT_TRUE(resolve_model_registration("MiniMaxH3Pipeline",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(resolved_name, "MiniMaxH3Pipeline");
  EXPECT_EQ(ModelRegistry::get_model_backend(resolved_name), "dit");
}

}  // namespace
}  // namespace xllm
