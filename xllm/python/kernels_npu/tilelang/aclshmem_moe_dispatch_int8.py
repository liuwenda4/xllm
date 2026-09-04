"""Fixed-capacity ACLSHMEM INT8 MoE Dispatch kernel."""

import tilelang.language as T


def build_aclshmem_moe_dispatch_int8_kernel(
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

    physical_hidden = max(32, (hidden_size + 31) // 32 * 32)
    global_experts = ep_world_size * local_experts
    max_capacity = ep_world_size * local_tokens * local_experts

    @T.prim_func
    def aclshmem_moe_dispatch_int8(
        payload: T.Tensor([local_tokens, physical_hidden], "int8"),
        scale: T.Tensor([local_tokens], "float32"),
        expert_ids: T.Tensor([local_tokens, topk], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        win_payload: T.Tensor([max_capacity, physical_hidden], "int8"),
        win_scale: T.Tensor([max_capacity, 8], "float32"),
        win_triplet: T.Tensor([max_capacity, 8], "int32"),
        win_status: T.Tensor([global_experts, 8], "int32"),
        win_credit: T.Tensor([global_experts, 8], "int32"),
        expand_payload: T.Tensor([max_capacity, physical_hidden], "int8"),
        expand_scale: T.Tensor([max_capacity], "float32"),
        expand_ids: T.Tensor([max_capacity, 3], "int32"),
        global_prefix: T.Tensor([global_experts], "int32"),
        expert_token_nums: T.Tensor([local_experts], "int64"),
        ep_receive_count: T.Tensor([local_experts], "int32"),
        active_mask: T.Tensor([max_capacity], "int32"),
        actual_count: T.Tensor([1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            payload_ub = T.alloc_ub([physical_hidden], "int8")
            scale_ub = T.alloc_ub([8], "float32")
            triplet_ub = T.alloc_ub([8], "int32")
            status_ub = T.alloc_ub([8], "int32")
            receive_status_ub = T.alloc_ub([8], "int32")
            zero_status_ub = T.alloc_ub([8], "int32")
            credit_ub = T.alloc_ub([8], "int32")
            expert_ids_ub = T.alloc_ub([local_tokens, 8], "int32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")
            occurrence_ub = T.alloc_ub([1], "int32")
            route_count_ub = T.alloc_ub([1], "int32")
            output_row_ub = T.alloc_ub([1], "int32")
            expert_start_ub = T.alloc_ub([1], "int32")

            with T.Scope("V"):
                if vid == 0:
                    T.copy(generation_id, generation_ub)
                    T.copy(iteration_id, iteration_ub)
                    T.tile.fill(scale_ub, 0.0)
                    T.tile.fill(triplet_ub, 0)
                    T.tile.fill(status_ub, 0)
                    T.tile.fill(zero_status_ub, 0)
                    T.tile.fill(credit_ub, 0)
                    for token_id in T.serial(local_tokens):
                        T.copy(
                            expert_ids[token_id, 0:topk],
                            expert_ids_ub[token_id, 0:8],
                        )
                    T.set_flag("mte2", "v", 3)
                    T.wait_flag("mte2", "v", 3)
                    for row in T.serial(max_capacity):
                        active_mask[row] = 0

                    if generation_ub[0] > 1:
                        for expert_id in T.serial(global_experts):
                            T.shmem_signal_wait_until(
                                win_credit,
                                expert_id * 8,
                                0,
                                generation_ub[0] - 1,
                            )

                    for token_id in T.serial(local_tokens):
                        for topk_slot in T.serial(topk):
                            expert_id = expert_ids_ub[token_id, topk_slot]
                            destination_rank = expert_id // local_experts
                            local_expert_id = expert_id % local_experts
                            occurrence_ub[0] = 0
                            for previous_token in T.serial(local_tokens):
                                if previous_token < token_id:
                                    for previous_slot in T.serial(topk):
                                        if expert_ids_ub[previous_token, previous_slot] == expert_id:
                                            occurrence_ub[0] = occurrence_ub[0] + 1
                            remote_slot = (
                                rank * local_tokens * local_experts + local_expert_id * local_tokens + occurrence_ub[0]
                            )
                            T.copy(payload[token_id, 0], payload_ub)
                            scale_ub[0] = scale[token_id]
                            triplet_ub[0] = rank
                            triplet_ub[1] = token_id
                            triplet_ub[2] = topk_slot
                            if destination_rank == rank:
                                T.copy(payload_ub, win_payload[remote_slot, 0])
                                T.copy(scale_ub, win_scale[remote_slot, 0])
                                T.copy(triplet_ub, win_triplet[remote_slot, 0])
                                T.set_flag("mte3", "s", 4)
                                T.wait_flag("mte3", "s", 4)
                            else:
                                T.shmem_ub_put_nbi(
                                    payload_ub,
                                    win_payload,
                                    physical_hidden,
                                    destination_rank,
                                    remote_slot * physical_hidden,
                                )
                                T.shmem_mte_quiet()
                                T.shmem_ub_put_nbi(
                                    scale_ub,
                                    win_scale,
                                    8,
                                    destination_rank,
                                    remote_slot * 8,
                                )
                                T.shmem_mte_quiet()
                                T.shmem_ub_put_nbi(
                                    triplet_ub,
                                    win_triplet,
                                    8,
                                    destination_rank,
                                    remote_slot * 8,
                                )
                                T.shmem_mte_quiet()

                    for expert_id in T.serial(global_experts):
                        route_count_ub[0] = 0
                        for token_id in T.serial(local_tokens):
                            for topk_slot in T.serial(topk):
                                if expert_ids_ub[token_id, topk_slot] == expert_id:
                                    route_count_ub[0] = route_count_ub[0] + 1
                        destination_rank = expert_id // local_experts
                        local_expert_id = expert_id % local_experts
                        status_row = local_expert_id * ep_world_size + rank
                        status_ub[0] = generation_ub[0]
                        status_ub[1] = route_count_ub[0]
                        status_ub[2] = iteration_ub[0]
                        if destination_rank == rank:
                            T.copy(status_ub, win_status[status_row, 0])
                            T.set_flag("mte3", "s", 5)
                            T.wait_flag("mte3", "s", 5)
                        else:
                            T.shmem_ub_put_nbi(
                                status_ub,
                                win_status,
                                8,
                                destination_rank,
                                status_row * 8,
                            )
                            T.shmem_mte_quiet()

                    output_row_ub[0] = 0
                    for local_expert_id in T.serial(local_experts):
                        expert_start_ub[0] = output_row_ub[0]
                        for source_rank in T.serial(ep_world_size):
                            status_row = local_expert_id * ep_world_size + source_rank
                            T.shmem_signal_wait_until(
                                win_status,
                                status_row * 8,
                                0,
                                generation_ub[0],
                            )
                            T.copy(win_status[status_row, 0], receive_status_ub)
                            T.set_flag("mte2", "v", 6)
                            T.wait_flag("mte2", "v", 6)
                            route_count_ub[0] = receive_status_ub[1]
                            for route_index in T.serial(local_tokens):
                                if route_index < route_count_ub[0]:
                                    source_slot = (
                                        source_rank * local_tokens * local_experts
                                        + local_expert_id * local_tokens
                                        + route_index
                                    )
                                    T.copy(win_payload[source_slot, 0], payload_ub)
                                    T.copy(win_scale[source_slot, 0], scale_ub)
                                    T.copy(win_triplet[source_slot, 0], triplet_ub)
                                    T.set_flag("mte2", "v", 7)
                                    T.wait_flag("mte2", "v", 7)
                                    T.set_flag("v", "mte3", 9)
                                    T.wait_flag("v", "mte3", 9)
                                    T.copy(
                                        payload_ub,
                                        expand_payload[output_row_ub[0], 0],
                                    )
                                    T.copy(
                                        scale_ub[0:1],
                                        expand_scale[output_row_ub[0] : output_row_ub[0] + 1],
                                    )
                                    T.copy(
                                        triplet_ub[0:3],
                                        expand_ids[output_row_ub[0], 0:3],
                                    )
                                    T.set_flag("mte3", "mte2", 10)
                                    T.wait_flag("mte3", "mte2", 10)
                                    active_mask[output_row_ub[0]] = 1
                                    output_row_ub[0] = output_row_ub[0] + 1
                            global_prefix[status_row] = output_row_ub[0]
                            T.copy(zero_status_ub, win_status[status_row, 0])
                            credit_ub[0] = generation_ub[0]
                            if source_rank == rank:
                                T.copy(
                                    credit_ub,
                                    win_credit[
                                        rank * local_experts + local_expert_id,
                                        0,
                                    ],
                                )
                                T.set_flag("mte3", "s", 11)
                                T.wait_flag("mte3", "s", 11)
                            else:
                                T.shmem_ub_put_nbi(
                                    credit_ub,
                                    win_credit,
                                    8,
                                    source_rank,
                                    (rank * local_experts + local_expert_id) * 8,
                                )
                                T.shmem_mte_quiet()
                        route_count_ub[0] = output_row_ub[0] - expert_start_ub[0]
                        expert_token_nums[local_expert_id] = route_count_ub[0]
                        ep_receive_count[local_expert_id] = route_count_ub[0]
                    actual_count[0] = output_row_ub[0]
                    T.set_flag("mte3", "s", 8)
                    T.wait_flag("mte3", "s", 8)
                    T.tile.datacachecleanandinvalid_experiment(expand_payload, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(expand_scale, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(expand_ids, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(global_prefix, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(
                        expert_token_nums,
                        "SINGLE_CACHE_LINE",
                        "CACHELINE_ALL",
                    )
                    T.tile.datacachecleanandinvalid_experiment(
                        ep_receive_count,
                        "SINGLE_CACHE_LINE",
                        "CACHELINE_ALL",
                    )
                    T.tile.datacachecleanandinvalid_experiment(active_mask, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(actual_count, "SINGLE_CACHE_LINE", "CACHELINE_ALL")

    return aclshmem_moe_dispatch_int8
