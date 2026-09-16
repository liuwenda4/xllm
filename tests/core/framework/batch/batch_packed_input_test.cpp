/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_input_builder.h"
#include "core/framework/batch/dit_batch.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/request/dit_request.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "core/runtime/params_utils.h"

namespace xllm {

namespace {

void expect_linear_state_cache_op_eq(const LinearStateCacheOp& actual,
                                     const LinearStateCacheOp& expected) {
  EXPECT_EQ(actual.linear_state_id, expected.linear_state_id);
  EXPECT_EQ(actual.reset_requested, expected.reset_requested);
  EXPECT_EQ(actual.restore_requested, expected.restore_requested);
  EXPECT_EQ(actual.restore_src_slot_id, expected.restore_src_slot_id);
}

DiTForwardInput make_condition_forward_input() {
  DiTForwardInput input;
  input.batch_size = 2;
  input.prompt_embeds = torch::arange(24, torch::kBFloat16).reshape({2, 3, 4});
  input.text_token_tags = torch::tensor({{0, 1, 0}, {1, 0, 1}}, torch::kInt64);
  input.condition_schemas = {"xllm.minimax_h3.text_conditioning/v1",
                             "xllm.minimax_h3.text_conditioning/v1"};
  input.condition_source_backends = {"official_hf", "xllm_native"};
  input.condition_manifest_jsons = {"{\"request\":1}", "{\"request\":2}"};
  return input;
}

void expect_condition_forward_input_eq(const DiTForwardInput& actual,
                                       const DiTForwardInput& expected) {
  EXPECT_EQ(actual.batch_size, expected.batch_size);
  ASSERT_TRUE(actual.prompt_embeds.defined());
  EXPECT_TRUE(torch::equal(actual.prompt_embeds, expected.prompt_embeds));
  ASSERT_TRUE(actual.text_token_tags.defined());
  EXPECT_EQ(actual.text_token_tags.scalar_type(), torch::kInt64);
  EXPECT_TRUE(torch::equal(actual.text_token_tags, expected.text_token_tags));
  EXPECT_EQ(actual.condition_schemas, expected.condition_schemas);
  EXPECT_EQ(actual.condition_source_backends,
            expected.condition_source_backends);
  EXPECT_EQ(actual.condition_manifest_jsons, expected.condition_manifest_jsons);
}

std::shared_ptr<DiTRequest> make_conditioned_dit_request(
    const std::string& request_id,
    const std::vector<int64_t>& tags,
    const std::string& source_backend,
    const std::string& manifest,
    DiTRequestKind request_kind = DiTRequestKind::kVideo) {
  DiTInputParams input_params;
  input_params.prompt_embed = torch::zeros({3, 5120}, torch::kBFloat16);
  input_params.text_token_tags = torch::tensor(tags, torch::kInt64);
  input_params.text_token_tags_is_set = true;
  input_params.condition_schema = "xllm.minimax_h3.text_conditioning/v1";
  input_params.condition_source_backend = source_backend;
  input_params.condition_manifest_json = manifest;

  DiTGenerationParams generation_params;
  DiTOutputFunc output_func = [](const DiTRequestOutput&) { return true; };
  DiTOutputsFunc outputs_func = [](const std::vector<DiTRequestOutput>&) {
    return std::vector<bool>{};
  };
  DiTRequestState state(
      input_params, generation_params, output_func, outputs_func, request_kind);
  return std::make_shared<DiTRequest>(request_id, "rid", "rtime", state);
}

}  // namespace

template <typename T>
bool tensor_equals_vector(const torch::Tensor& tensor,
                          const std::vector<T>& values) {
  auto flat = tensor.flatten();
  if (flat.size(0) != values.size()) {
    return false;
  }
  for (int64_t i = 0; i < flat.size(0); ++i) {
    if (flat[i].item<T>() != values[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackPreservesLinearStateCacheOps) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  LinearStateCacheOp restore_op;
  restore_op.linear_state_id = 7;
  restore_op.restore_requested = true;
  restore_op.restore_src_slot_id = 3;

  LinearStateCacheOp no_restore_op;
  no_restore_op.linear_state_id = 8;
  no_restore_op.reset_requested = true;

  input.input_params.linear_state_cache_ops = {restore_op, no_restore_op};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_params.linear_state_cache_ops.empty());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_EQ(unpacked_input.input_params.linear_state_cache_ops.size(), 2u);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[0], restore_op);
  expect_linear_state_cache_op_eq(
      unpacked_input.input_params.linear_state_cache_ops[1], no_restore_op);
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackRestoresSampleIdxes) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  ASSERT_TRUE(input.sampling_params.sample_idxes.defined());
  input.sampling_params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0x5)}},
                    torch::TensorOptions().dtype(torch::kInt32));

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_FALSE(lazy_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_TRUE(unpacked_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      unpacked_input.sampling_params.sample_idxes, {0}));
  ASSERT_TRUE(unpacked_input.sampling_params.filter_bitmask.defined());
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.filter_bitmask,
                           input.sampling_params.filter_bitmask));
}

TEST(BatchPackedInputTest, PackedProtoLazyToPreservesJsonMetadata) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;

  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{1, 2, 3, 4},
                    torch::Tensor(),
                    mm_data,
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&sequence};
  std::vector<uint32_t> allowed_max_tokens = {4};
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/2,
                            nullptr,
                            BatchForwardType::DECODE);
  ForwardInput input = builder.build_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0);

  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));
  input.json_object_state_snapshots = {state.snapshot()};
  input.sample_sequence_ids = {"req-json#0"};
  input.sample_prior_output_rows = {-1};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);
  EXPECT_TRUE(lazy_input.json_object_state_snapshots.empty());

  const ForwardInput materialized_input =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ASSERT_EQ(materialized_input.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(materialized_input.json_object_state_snapshots[0].token_ids,
            std::vector<int32_t>({0}));
  EXPECT_FALSE(
      materialized_input.json_object_state_snapshots[0].reasoning_enabled);
  EXPECT_EQ(materialized_input.sample_sequence_ids,
            std::vector<std::string>({"req-json#0"}));
  EXPECT_EQ(materialized_input.sample_prior_output_rows,
            std::vector<int32_t>({-1}));
}

TEST(BatchPackedInputTest, WorkerProtoPreservesMiniMaxH3ConditionBundle) {
  const DiTForwardInput expected = make_condition_forward_input();
  proto::DiTForwardInput serialized_proto;
  ASSERT_TRUE(dit_forward_input_to_proto(expected, &serialized_proto));

  const std::string wire = serialized_proto.SerializeAsString();
  proto::DiTForwardInput parsed_proto;
  ASSERT_TRUE(parsed_proto.ParseFromString(wire));

  DiTForwardInput actual;
  ASSERT_TRUE(proto_to_dit_forward_input(parsed_proto, actual));
  expect_condition_forward_input_eq(actual, expected);
}

TEST(BatchPackedInputTest, SharedMemoryPreservesMiniMaxH3ConditionBundle) {
  ForwardInput input;
  const DiTForwardInput expected = make_condition_forward_input();
  input.input_params.dit_forward_input = expected;

  bool is_creator = false;
  const std::string shm_name = ForwardSharedMemoryManager::create_unique_name(
      "batch_packed_h3_condition",
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);

  ASSERT_TRUE(writer_manager.input_write(input));
  ForwardInput round_trip;
  reader_manager.input_read(round_trip, torch::Device(torch::kCPU));

  expect_condition_forward_input_eq(round_trip.input_params.dit_forward_input,
                                    expected);
}

TEST(BatchPackedInputTest, EncodedMediaFlowsThroughRequestOutput) {
  std::shared_ptr<DiTRequest> request = make_conditioned_dit_request(
      "encoded-media", {0, 1, 0}, "official_hf", "{\"request\":1}");
  request->state().generation_params().seed = 1234;
  request->state().generation_params().seed_is_set = true;
  DiTBatch batch;
  batch.add(request);

  DiTEncodedMedia media;
  media.data = std::string("ftyp\0mdat", 9);
  media.mime_type = "video/mp4";
  media.container = "mp4";
  media.width = 1344;
  media.height = 768;
  media.num_frames = 124;
  media.fps = 24.0;
  media.audio_sample_rate = 32000;
  media.audio_channels = 2;
  DiTForwardOutput forward_output;
  forward_output.encoded_media.push_back(media);

  batch.process_forward_output(forward_output);
  const DiTRequestOutput output = request->generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  const DiTGenerationOutput& actual = output.outputs.front();
  EXPECT_EQ(actual.image, media.data);
  EXPECT_EQ(actual.mime_type, media.mime_type);
  EXPECT_EQ(actual.container, media.container);
  EXPECT_EQ(actual.width, media.width);
  EXPECT_EQ(actual.height, media.height);
  EXPECT_EQ(actual.num_frames, media.num_frames);
  EXPECT_DOUBLE_EQ(actual.video_fps, media.fps);
  EXPECT_EQ(actual.audio_sample_rate, media.audio_sample_rate);
  EXPECT_EQ(actual.audio_channels, media.audio_channels);
  EXPECT_TRUE(actual.seed_is_set);
  EXPECT_EQ(actual.seed, 1234);
}

TEST(BatchPackedInputTest, MultipleEncodedMediaMapToOneRequest) {
  std::shared_ptr<DiTRequest> request = make_conditioned_dit_request(
      "encoded-media-multi", {0, 1, 0}, "official_hf", "{}");
  request->state().generation_params().num_videos_per_prompt = 2;
  DiTBatch batch;
  batch.add(request);

  DiTEncodedMedia first;
  first.data = "first";
  first.mime_type = "video/mp4";
  first.container = "mp4";
  DiTEncodedMedia second = first;
  second.data = "second";
  DiTForwardOutput forward_output;
  forward_output.encoded_media = {first, second};

  batch.process_forward_output(forward_output);
  const DiTRequestOutput output = request->generate_output();

  ASSERT_EQ(output.outputs.size(), 2u);
  EXPECT_EQ(output.outputs[0].image, "first");
  EXPECT_EQ(output.outputs[0].index, 0u);
  EXPECT_EQ(output.outputs[1].image, "second");
  EXPECT_EQ(output.outputs[1].index, 1u);
}

TEST(BatchPackedInputTest, EncodedMediaUsesRequestMajorOrdering) {
  std::shared_ptr<DiTRequest> first_request = make_conditioned_dit_request(
      "encoded-media-first", {0, 1, 0}, "official_hf", "{}");
  std::shared_ptr<DiTRequest> second_request = make_conditioned_dit_request(
      "encoded-media-second", {1, 0, 1}, "official_hf", "{}");
  first_request->state().generation_params().num_videos_per_prompt = 2;
  second_request->state().generation_params().num_videos_per_prompt = 2;
  DiTBatch batch;
  batch.add(first_request);
  batch.add(second_request);

  DiTForwardOutput forward_output;
  for (const std::string& data :
       {"first-0", "first-1", "second-0", "second-1"}) {
    DiTEncodedMedia media;
    media.data = data;
    media.mime_type = "video/mp4";
    media.container = "mp4";
    forward_output.encoded_media.emplace_back(std::move(media));
  }

  batch.process_forward_output(forward_output);
  const DiTRequestOutput first_output = first_request->generate_output();
  const DiTRequestOutput second_output = second_request->generate_output();

  ASSERT_EQ(first_output.outputs.size(), 2u);
  EXPECT_EQ(first_output.outputs[0].image, "first-0");
  EXPECT_EQ(first_output.outputs[1].image, "first-1");
  ASSERT_EQ(second_output.outputs.size(), 2u);
  EXPECT_EQ(second_output.outputs[0].image, "second-0");
  EXPECT_EQ(second_output.outputs[1].image, "second-1");
}

TEST(BatchPackedInputTest, EncodedAudioUsesAudioOutputField) {
  std::shared_ptr<DiTRequest> request = make_conditioned_dit_request(
      "encoded-audio", {0, 1, 0}, "official_hf", "{}", DiTRequestKind::kAudio);
  DiTBatch batch;
  batch.add(request);

  DiTEncodedMedia media;
  media.data = std::string("RIFF\0WAVE", 9);
  media.mime_type = "audio/wav";
  media.container = "wav";
  media.audio_sample_rate = 32000;
  media.audio_channels = 2;
  DiTForwardOutput forward_output;
  forward_output.encoded_media.push_back(media);

  batch.process_forward_output(forward_output);
  const DiTRequestOutput output = request->generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_EQ(output.outputs[0].audio, media.data);
  EXPECT_TRUE(output.outputs[0].image.empty());
  EXPECT_EQ(output.outputs[0].mime_type, "audio/wav");
  EXPECT_EQ(output.outputs[0].audio_sample_rate, 32000);
  EXPECT_EQ(output.outputs[0].audio_channels, 2);
}

TEST(BatchPackedInputTest, DiTBatchStacksMiniMaxH3TokenTags) {
  DiTBatch batch;
  batch.add(make_conditioned_dit_request(
      "request-1", {0, 1, 0}, "official_hf", "{\"request\":1}"));
  batch.add(make_conditioned_dit_request(
      "request-2", {1, 0, 1}, "xllm_native", "{\"request\":2}"));

  const DiTForwardInput input = batch.prepare_forward_input();

  ASSERT_TRUE(input.prompt_embeds.defined());
  EXPECT_EQ(input.prompt_embeds.sizes(), (torch::IntArrayRef{2, 3, 5120}));
  ASSERT_TRUE(input.text_token_tags.defined());
  EXPECT_EQ(input.text_token_tags.sizes(), (torch::IntArrayRef{2, 3}));
  EXPECT_TRUE(
      torch::equal(input.text_token_tags,
                   torch::tensor({{0, 1, 0}, {1, 0, 1}}, torch::kInt64)));
  EXPECT_EQ(input.condition_schemas,
            (std::vector<std::string>{"xllm.minimax_h3.text_conditioning/v1",
                                      "xllm.minimax_h3.text_conditioning/v1"}));
  EXPECT_EQ(input.condition_source_backends,
            (std::vector<std::string>{"official_hf", "xllm_native"}));
  EXPECT_EQ(input.condition_manifest_jsons,
            (std::vector<std::string>{"{\"request\":1}", "{\"request\":2}"}));
}

}  // namespace xllm
