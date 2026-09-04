"""Fixed-output INT8 dequantization before the ACLSHMEM MoE expert stage."""

import tilelang.language as T


def build_aclshmem_moe_dequant_int8_kernel(
    *,
    max_capacity: int,
    hidden_size: int,
):
    if max_capacity <= 0 or hidden_size <= 0:
        raise ValueError("max_capacity and hidden_size must be positive")

    physical_hidden = max(32, (hidden_size + 31) // 32 * 32)
    rows_per_core = 2
    core_count = (max_capacity + rows_per_core - 1) // rows_per_core

    @T.prim_func
    def aclshmem_moe_dequant_int8(
        payload: T.Tensor([max_capacity, physical_hidden], "int8"),
        scale: T.Tensor([max_capacity], "float32"),
        active_mask: T.Tensor([max_capacity], "int32"),
        output: T.Tensor([max_capacity, hidden_size], "bfloat16"),
    ):
        with T.Kernel(core_count, is_npu=True) as (cid, vid):
            row = cid * rows_per_core + vid
            payload_ub = T.alloc_ub([physical_hidden], "int8")
            payload_fp16_ub = T.alloc_ub([physical_hidden], "float16")
            payload_fp32_ub = T.alloc_ub([physical_hidden], "float32")
            output_ub = T.alloc_ub([physical_hidden], "bfloat16")

            with T.Scope("V"):
                if row < max_capacity:
                    if active_mask[row] != 0:
                        T.copy(payload[row, 0], payload_ub)
                        T.tile.cast(
                            payload_fp16_ub,
                            payload_ub,
                            mode="CAST_NONE",
                            count=physical_hidden,
                        )
                        T.pipe_barrier("v")
                        T.tile.cast(
                            payload_fp32_ub,
                            payload_fp16_ub,
                            mode="CAST_NONE",
                            count=physical_hidden,
                        )
                        T.pipe_barrier("v")
                        T.tile.mul(
                            payload_fp32_ub,
                            payload_fp32_ub,
                            scale[row],
                        )
                        T.pipe_barrier("v")
                        T.tile.cast(
                            output_ub,
                            payload_fp32_ub,
                            mode="CAST_RINT",
                            count=physical_hidden,
                        )
                    else:
                        T.tile.fill(output_ub, 0)
                    T.copy(output_ub[0:hidden_size], output[row, 0:hidden_size])

    return aclshmem_moe_dequant_int8
