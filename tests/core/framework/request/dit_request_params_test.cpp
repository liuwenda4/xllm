/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/dit_request_params.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "text_generation.pb.h"
#include "video_generation.pb.h"

namespace xllm {
namespace {

// Helper to build a minimal TextGenerationRequest.
proto::TextGenerationRequest MakeTextRequest(const std::string& model,
                                             const std::string& prompt) {
  proto::TextGenerationRequest request;
  request.set_model(model);
  auto* input = request.mutable_input();
  input->set_prompt(prompt);
  return request;
}

constexpr char kConditionSchema[] = "xllm.minimax_h3.text_conditioning/v1";

void SetBFloat16Tensor(proto::Tensor* tensor,
                       int64_t tokens,
                       int64_t hidden_size = 5120) {
  tensor->set_datatype("BF16");
  tensor->add_shape(tokens);
  tensor->add_shape(hidden_size);
  tensor->mutable_contents()->set_bytes_contents(
      std::string(tokens * hidden_size * sizeof(torch::BFloat16), '\0'));
}

void SetInt64Tensor(proto::Tensor* tensor, const std::vector<int64_t>& values) {
  tensor->set_datatype("INT64");
  tensor->add_shape(values.size());
  for (int64_t value : values) {
    tensor->mutable_contents()->add_int64_contents(value);
  }
}

std::string MakeConditionManifest(const std::string& source_backend,
                                  int64_t token_count) {
  return nlohmann::json({{"schema", kConditionSchema},
                         {"source_backend", source_backend},
                         {"decoder_layer_index", 49},
                         {"hidden_state_slot", 50},
                         {"token_count", token_count},
                         {"hidden_digest", std::string(64, 'a')},
                         {"token_tags_digest", std::string(64, 'b')}})
      .dump();
}

proto::VideoGenerationRequest MakeVideoRequest() {
  proto::VideoGenerationRequest request;
  request.set_model("MiniMax-H3");
  request.mutable_input()->set_prompt("a test video");
  return request;
}

proto::VideoGenerationRequest MakeConditionedVideoRequest(
    const std::string& source_backend = "official_hf") {
  constexpr int64_t kTokens = 2;
  proto::VideoGenerationRequest request;
  request.set_model("MiniMax-H3");
  auto* input = request.mutable_input();
  SetBFloat16Tensor(input->mutable_prompt_embed(), kTokens);
  SetInt64Tensor(input->mutable_text_token_tags(), {0, 1});
  input->set_condition_schema(kConditionSchema);
  input->set_condition_source_backend(source_backend);
  input->set_condition_manifest_json(
      MakeConditionManifest(source_backend, kTokens));
  return request;
}

void ExpectInvalidVideoRequest(const proto::VideoGenerationRequest& request) {
  DiTRequestParams params(request, "rid", "rtime");
  bool callback_called = false;
  StatusCode error_code = StatusCode::OK;
  const bool valid = params.verify_params([&](DiTRequestOutput output) {
    callback_called = true;
    if (output.status.has_value()) {
      error_code = output.status->code();
    }
    return true;
  });

  EXPECT_FALSE(valid);
  EXPECT_TRUE(callback_called);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
}

// ===========================================================================
// DiTRequestParams constructor from TextGenerationRequest
// ===========================================================================

TEST(DiTRequestParamsTest, TextRequestSetsKindToText) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_kind, DiTRequestKind::kText);
}

TEST(DiTRequestParamsTest, TextRequestMapsModel) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.model, "Cola-DLM");
}

TEST(DiTRequestParamsTest, TextRequestMapsPrompt) {
  auto req = MakeTextRequest("Cola-DLM", "hello world");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.input_params.prompt, "hello world");
}

TEST(DiTRequestParamsTest, TextRequestUsesProvidedRequestId) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  req.set_request_id("custom-id");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_id, "custom-id");
}

TEST(DiTRequestParamsTest, TextRequestGeneratesRequestIdWhenAbsent) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_FALSE(params.request_id.empty());
  EXPECT_EQ(params.request_id.substr(0, 8), "textgen-");
}

TEST(DiTRequestParamsTest, TextRequestMapsXRequestIdAndTime) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "xrid-123", "xrtime-456");

  EXPECT_EQ(params.x_request_id, "xrid-123");
  EXPECT_EQ(params.x_request_time, "xrtime-456");
}

TEST(DiTRequestParamsTest, TextRequestDefaultGenerationParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  // Default values when no parameters set.
  EXPECT_FALSE(params.generation_params.seed_is_set);
  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
}

TEST(DiTRequestParamsTest, TextRequestMapsSeedAndSetsFlag) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_seed(42);
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.seed, 42);
  EXPECT_TRUE(params.generation_params.seed_is_set);
}

TEST(DiTRequestParamsTest, TextRequestMapsAllSamplingParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(32);
  p->set_guidance_scale(5.0f);
  p->set_temperature(0.8f);
  p->set_top_k(50);
  p->set_top_p(0.95f);
  p->set_repetition_penalty(1.2f);

  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 128);
  EXPECT_EQ(params.generation_params.diffusion_steps, 32);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 5.0f);
  EXPECT_FLOAT_EQ(params.generation_params.temperature, 0.8f);
  EXPECT_EQ(params.generation_params.top_k, 50);
  EXPECT_FLOAT_EQ(params.generation_params.top_p, 0.95f);
  EXPECT_FLOAT_EQ(params.generation_params.repetition_penalty, 1.2f);
}

TEST(DiTRequestParamsTest, TextRequestWithoutParametersUsesDefaults) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  // Don't set parameters at all.
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
  EXPECT_FALSE(params.generation_params.seed_is_set);
}

// ===========================================================================
// verify_params for kText
// ===========================================================================

TEST(DiTRequestParamsVerifyTest, TextValidParamsReturnsTrue) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  bool called = false;
  auto result = params.verify_params([&](DiTRequestOutput /*unused*/) -> bool {
    called = true;
    return true;
  });

  EXPECT_TRUE(result);
  EXPECT_FALSE(called);  // No error callback on success.
}

TEST(DiTRequestParamsVerifyTest, TextEmptyPromptReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "");
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  std::string error_msg;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
      error_msg = output.status.value().message();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(error_msg, "prompt is empty");
}

TEST(DiTRequestParamsVerifyTest, TextEmptyModelReturnsFalse) {
  auto req = MakeTextRequest("", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  std::string error_msg;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
      error_msg = output.status.value().message();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(error_msg, "model is empty");
}

TEST(DiTRequestParamsVerifyTest, TextZeroMaxNewTokensReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(0);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
}

TEST(DiTRequestParamsVerifyTest, TextNegativeDiffusionStepsReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(-1);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
}

// ===========================================================================
// Video MiniMax-H3 condition bundle parsing and validation
// ===========================================================================

TEST(DiTRequestParamsTest, VideoRequestParsesMiniMaxH3ConditionBundle) {
  const auto request = MakeConditionedVideoRequest("xllm_native");
  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_EQ(params.request_kind, DiTRequestKind::kVideo);
  ASSERT_TRUE(params.input_params.prompt_embed.defined());
  EXPECT_EQ(params.input_params.prompt_embed.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(params.input_params.prompt_embed.sizes(),
            (torch::IntArrayRef{2, 5120}));
  ASSERT_TRUE(params.input_params.text_token_tags.defined());
  EXPECT_TRUE(torch::equal(params.input_params.text_token_tags,
                           torch::tensor({0, 1}, torch::kInt64)));
  ASSERT_TRUE(params.input_params.condition_schema.has_value());
  EXPECT_EQ(*params.input_params.condition_schema, kConditionSchema);
  ASSERT_TRUE(params.input_params.condition_source_backend.has_value());
  EXPECT_EQ(*params.input_params.condition_source_backend, "xllm_native");
  ASSERT_TRUE(params.input_params.condition_manifest_json.has_value());
  EXPECT_EQ(*params.input_params.condition_manifest_json,
            request.input().condition_manifest_json());
}

TEST(DiTRequestParamsVerifyTest, GenericVideoRemainsValidWithoutBundle) {
  const auto request = MakeVideoRequest();
  DiTRequestParams params(request, "rid", "rtime");
  bool callback_called = false;

  EXPECT_TRUE(params.verify_params([&](DiTRequestOutput) {
    callback_called = true;
    return true;
  }));
  EXPECT_FALSE(callback_called);
}

TEST(DiTRequestParamsVerifyTest, CompleteMiniMaxH3BundlesAreValid) {
  for (const std::string backend : {"official_hf", "xllm_native"}) {
    const auto request = MakeConditionedVideoRequest(backend);
    DiTRequestParams params(request, "rid", "rtime");
    bool callback_called = false;

    EXPECT_TRUE(params.verify_params([&](DiTRequestOutput) {
      callback_called = true;
      return true;
    }));
    EXPECT_FALSE(callback_called);
  }
}

TEST(DiTRequestParamsVerifyTest, PartialMiniMaxH3BundleFailsClosed) {
  auto request = MakeVideoRequest();
  request.mutable_input()->set_condition_schema(kConditionSchema);

  ExpectInvalidVideoRequest(request);

  auto malformed_tags_only = MakeVideoRequest();
  malformed_tags_only.mutable_input()->mutable_text_token_tags();
  ExpectInvalidVideoRequest(malformed_tags_only);

  auto unknown_tags_datatype = MakeVideoRequest();
  unknown_tags_datatype.mutable_input()
      ->mutable_text_token_tags()
      ->set_datatype("UNKNOWN");
  ExpectInvalidVideoRequest(unknown_tags_datatype);
}

TEST(DiTRequestParamsVerifyTest, RejectsUnsupportedConditionIdentity) {
  auto wrong_schema = MakeConditionedVideoRequest();
  wrong_schema.mutable_input()->set_condition_schema("unsupported/v1");
  ExpectInvalidVideoRequest(wrong_schema);

  auto wrong_backend = MakeConditionedVideoRequest();
  wrong_backend.mutable_input()->set_condition_source_backend("unknown");
  ExpectInvalidVideoRequest(wrong_backend);
}

TEST(DiTRequestParamsVerifyTest, RejectsInvalidConditionTensorAbi) {
  auto unknown_prompt_datatype = MakeConditionedVideoRequest();
  unknown_prompt_datatype.mutable_input()->mutable_prompt_embed()->set_datatype(
      "UNKNOWN");
  ExpectInvalidVideoRequest(unknown_prompt_datatype);

  auto wrong_prompt_dtype = MakeConditionedVideoRequest();
  DiTRequestParams wrong_prompt_dtype_params(
      wrong_prompt_dtype, "rid", "rtime");
  wrong_prompt_dtype_params.input_params.prompt_embed =
      torch::zeros({2, 5120}, torch::kFloat32);
  bool callback_called = false;
  EXPECT_FALSE(wrong_prompt_dtype_params.verify_params([&](DiTRequestOutput) {
    callback_called = true;
    return true;
  }));
  EXPECT_TRUE(callback_called);

  auto wrong_prompt_shape = MakeConditionedVideoRequest();
  auto* prompt_embed =
      wrong_prompt_shape.mutable_input()->mutable_prompt_embed();
  prompt_embed->clear_shape();
  prompt_embed->add_shape(2);
  prompt_embed->add_shape(2560);
  prompt_embed->mutable_contents()->set_bytes_contents(
      std::string(2 * 2560 * sizeof(torch::BFloat16), '\0'));
  ExpectInvalidVideoRequest(wrong_prompt_shape);

  auto wrong_tags_dtype = MakeConditionedVideoRequest();
  auto* tags = wrong_tags_dtype.mutable_input()->mutable_text_token_tags();
  tags->set_datatype("INT32");
  tags->mutable_contents()->clear_int64_contents();
  tags->mutable_contents()->add_int_contents(0);
  tags->mutable_contents()->add_int_contents(1);
  ExpectInvalidVideoRequest(wrong_tags_dtype);

  auto noncontiguous_prompt = MakeConditionedVideoRequest();
  DiTRequestParams params(noncontiguous_prompt, "rid", "rtime");
  params.input_params.prompt_embed =
      torch::zeros({5120, 2}, torch::kBFloat16).transpose(0, 1);
  ASSERT_FALSE(params.input_params.prompt_embed.is_contiguous());
  callback_called = false;
  EXPECT_FALSE(params.verify_params([&](DiTRequestOutput) {
    callback_called = true;
    return true;
  }));
  EXPECT_TRUE(callback_called);
}

TEST(DiTRequestParamsVerifyTest, RejectsInvalidTextTokenTags) {
  auto mismatched_count = MakeConditionedVideoRequest();
  auto* tags = mismatched_count.mutable_input()->mutable_text_token_tags();
  tags->clear_shape();
  tags->add_shape(1);
  tags->mutable_contents()->clear_int64_contents();
  tags->mutable_contents()->add_int64_contents(0);
  ExpectInvalidVideoRequest(mismatched_count);

  auto invalid_value = MakeConditionedVideoRequest();
  tags = invalid_value.mutable_input()->mutable_text_token_tags();
  tags->mutable_contents()->set_int64_contents(1, 2);
  ExpectInvalidVideoRequest(invalid_value);

  auto noncontiguous_tags = MakeConditionedVideoRequest();
  DiTRequestParams params(noncontiguous_tags, "rid", "rtime");
  params.input_params.text_token_tags =
      torch::zeros({2, 2}, torch::kInt64).select(1, 0);
  ASSERT_FALSE(params.input_params.text_token_tags.is_contiguous());
  bool callback_called = false;
  EXPECT_FALSE(params.verify_params([&](DiTRequestOutput) {
    callback_called = true;
    return true;
  }));
  EXPECT_TRUE(callback_called);
}

TEST(DiTRequestParamsVerifyTest, RejectsInvalidConditionManifest) {
  auto invalid_json = MakeConditionedVideoRequest();
  invalid_json.mutable_input()->set_condition_manifest_json("not-json");
  ExpectInvalidVideoRequest(invalid_json);

  auto non_object = MakeConditionedVideoRequest();
  non_object.mutable_input()->set_condition_manifest_json("[]");
  ExpectInvalidVideoRequest(non_object);

  auto mismatched_fields = MakeConditionedVideoRequest();
  nlohmann::json manifest = nlohmann::json::parse(
      mismatched_fields.input().condition_manifest_json());
  manifest["token_count"] = 3;
  mismatched_fields.mutable_input()->set_condition_manifest_json(
      manifest.dump());
  ExpectInvalidVideoRequest(mismatched_fields);

  auto non_string_digest = MakeConditionedVideoRequest();
  manifest = nlohmann::json::parse(
      non_string_digest.input().condition_manifest_json());
  manifest["hidden_digest"] = 123;
  non_string_digest.mutable_input()->set_condition_manifest_json(
      manifest.dump());
  ExpectInvalidVideoRequest(non_string_digest);

  auto short_digest = MakeConditionedVideoRequest();
  manifest =
      nlohmann::json::parse(short_digest.input().condition_manifest_json());
  manifest["token_tags_digest"] = "abc";
  short_digest.mutable_input()->set_condition_manifest_json(manifest.dump());
  ExpectInvalidVideoRequest(short_digest);
}

// ===========================================================================
// DiTGenerationParams equality with new text fields
// ===========================================================================

TEST(DiTGenerationParamsEqualityTest, EqualParamsAreEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;

  EXPECT_EQ(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentMaxNewTokensAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.max_new_tokens = 128;
  b.max_new_tokens = 256;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentSeedIsSetAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.seed_is_set = true;
  b.seed_is_set = false;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentDiffusionStepsAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.diffusion_steps = 16;
  b.diffusion_steps = 32;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTemperatureAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.temperature = 0.0f;
  b.temperature = 1.0f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopKAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_k = 0;
  b.top_k = 50;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopPAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_p = 1.0f;
  b.top_p = 0.9f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentRepetitionPenaltyAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.repetition_penalty = 1.0f;
  b.repetition_penalty = 1.1f;

  EXPECT_NE(a, b);
}

}  // namespace
}  // namespace xllm
