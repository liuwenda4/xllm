"""Fixed-capacity BF16 ACLSHMEM MoE Combine kernel."""

import tilelang.language as T


def build_aclshmem_moe_combine_bf16_kernel(
    *,
    local_tokens: int,
    hidden_size: int,
    topk: int,
    ep_world_size: int,
    local_experts: int,
    rank: int,
):
    if local_tokens <= 0 or hidden_size <= 0:
        raise ValueError("local_tokens and hidden_size must be positive")
    if ep_world_size <= 1 or local_experts <= 0:
        raise ValueError("ep_world_size and local_experts must be valid")
    if not 0 <= rank < ep_world_size:
        raise ValueError("rank is outside ep_world_size")
    if topk <= 0 or topk > 8 or topk > ep_world_size * local_experts:
        raise ValueError("topk is unsupported")

    max_capacity = ep_world_size * local_tokens * local_experts
    combine_rows = local_tokens * topk
    physical_hidden = (hidden_size + 15) // 16 * 16
    credit_rows = ep_world_size * combine_rows

    @T.prim_func
    def aclshmem_moe_combine_bf16(
        expert_output: T.Tensor([max_capacity, hidden_size], "bfloat16"),
        expand_ids: T.Tensor([max_capacity, 3], "int32"),
        active_mask: T.Tensor([max_capacity], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        route_weights: T.Tensor([local_tokens, topk], "float32"),
        win_payload: T.Tensor([combine_rows, physical_hidden], "bfloat16"),
        win_status: T.Tensor([combine_rows, 8], "int32"),
        win_credit: T.Tensor([credit_rows, 8], "int32"),
        output: T.Tensor([local_tokens, hidden_size], "bfloat16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            payload_ub = T.alloc_ub([physical_hidden], "bfloat16")
            payload_fp32_ub = T.alloc_ub([physical_hidden], "float32")
            accumulator_ub = T.alloc_ub([physical_hidden], "float32")
            output_ub = T.alloc_ub([physical_hidden], "bfloat16")
            triplet_ub = T.alloc_ub([8], "int32")
            status_ub = T.alloc_ub([8], "int32")
            zero_status_ub = T.alloc_ub([8], "int32")
            credit_ub = T.alloc_ub([8], "int32")
            weights_ub = T.alloc_ub([8], "float32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")

            with T.Scope("V"):
                if vid == 0:
                    T.copy(generation_id, generation_ub)
                    T.copy(iteration_id, iteration_ub)
                    T.tile.fill(status_ub, 0)
                    T.tile.fill(zero_status_ub, 0)
                    T.tile.fill(credit_ub, 0)

                    for row in T.serial(max_capacity):
                        if active_mask[row] != 0:
                            T.tile.fill(triplet_ub, 0)
                            T.copy(expand_ids[row, 0:3], triplet_ub[0:3])
                            owner_rank = triplet_ub[0]
                            token_id = triplet_ub[1]
                            topk_slot = triplet_ub[2]
                            combine_slot = token_id * topk + topk_slot
                            credit_row = owner_rank * combine_rows + combine_slot
                            if generation_ub[0] > 1:
                                T.shmem_signal_wait_until(
                                    win_credit,
                                    credit_row * 8,
                                    0,
                                    generation_ub[0] - 1,
                                )

                            T.copy(
                                expert_output[row, 0:hidden_size],
                                payload_ub,
                                pad_value=0,
                            )
                            status_ub[0] = generation_ub[0]
                            status_ub[1] = 1
                            status_ub[2] = rank
                            status_ub[3] = iteration_ub[0]
                            if owner_rank == rank:
                                T.copy(payload_ub, win_payload[combine_slot, 0])
                                T.copy(status_ub, win_status[combine_slot, 0])
                                T.set_flag("mte3", "s", 0)
                                T.wait_flag("mte3", "s", 0)
                            else:
                                T.shmem_ub_put_nbi(
                                    payload_ub,
                                    win_payload,
                                    physical_hidden,
                                    owner_rank,
                                    combine_slot * physical_hidden,
                                )
                                T.shmem_mte_quiet()
                                T.shmem_ub_put_nbi(
                                    status_ub,
                                    win_status,
                                    8,
                                    owner_rank,
                                    combine_slot * 8,
                                )
                                T.shmem_mte_quiet()

                    for token_id in T.serial(local_tokens):
                        T.tile.fill(accumulator_ub, 0.0)
                        T.tile.fill(weights_ub, 0.0)
                        T.copy(
                            route_weights[token_id, 0:topk],
                            weights_ub[0:topk],
                        )
                        for topk_slot in T.serial(topk):
                            combine_slot = token_id * topk + topk_slot
                            T.shmem_signal_wait_until(
                                win_status,
                                combine_slot * 8,
                                0,
                                generation_ub[0],
                            )
                            T.copy(win_payload[combine_slot, 0], payload_ub)
                            T.tile.cast(
                                payload_fp32_ub,
                                payload_ub,
                                mode="CAST_NONE",
                                count=physical_hidden,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                payload_fp32_ub,
                                payload_fp32_ub,
                                weights_ub[topk_slot],
                            )
                            T.pipe_barrier("v")
                            T.tile.add(
                                accumulator_ub,
                                accumulator_ub,
                                payload_fp32_ub,
                            )
                            T.pipe_barrier("v")

                        T.tile.cast(
                            output_ub,
                            accumulator_ub,
                            mode="CAST_RINT",
                            count=physical_hidden,
                        )
                        T.copy(
                            output_ub[0:hidden_size],
                            output[token_id, 0:hidden_size],
                        )
                        for topk_slot in T.serial(topk):
                            combine_slot = token_id * topk + topk_slot
                            T.copy(zero_status_ub, win_status[combine_slot, 0])
                            T.set_flag("mte3", "s", 1)
                            T.wait_flag("mte3", "s", 1)

                        credit_ub[0] = generation_ub[0]
                        for topk_slot in T.serial(topk):
                            combine_slot = token_id * topk + topk_slot
                            credit_row = rank * combine_rows + combine_slot
                            for producer_rank in T.serial(ep_world_size):
                                if producer_rank == rank:
                                    T.copy(
                                        credit_ub,
                                        win_credit[credit_row, 0],
                                    )
                                    T.set_flag("mte3", "s", 2)
                                    T.wait_flag("mte3", "s", 2)
                                else:
                                    T.shmem_ub_put_nbi(
                                        credit_ub,
                                        win_credit,
                                        8,
                                        producer_rank,
                                        credit_row * 8,
                                    )
                                    T.shmem_mte_quiet()

                    T.tile.datacachecleanandinvalid_experiment(output, "SINGLE_CACHE_LINE", "CACHELINE_ALL")

    return aclshmem_moe_combine_bf16
