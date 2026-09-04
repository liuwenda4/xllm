"""Fixed-output per-token INT8 quantization for ACLSHMEM MoE Dispatch."""

import tilelang.language as T


def build_aclshmem_moe_quant_int8_kernel(
    *,
    local_tokens: int,
    hidden_size: int,
):
    if local_tokens <= 0 or hidden_size <= 0:
        raise ValueError("local_tokens and hidden_size must be positive")

    physical_hidden = max(32, (hidden_size + 31) // 32 * 32)
    rows_per_core = 2
    core_count = (local_tokens + rows_per_core - 1) // rows_per_core

    @T.prim_func
    def aclshmem_moe_quant_int8(
        input: T.Tensor([local_tokens, hidden_size], "bfloat16"),
        payload: T.Tensor([local_tokens, physical_hidden], "int8"),
        scale: T.Tensor([local_tokens], "float32"),
    ):
        with T.Kernel(core_count, is_npu=True) as (cid, vid):
            row = cid * rows_per_core + vid
            input_ub = T.alloc_ub([physical_hidden], "bfloat16")
            input_fp32_ub = T.alloc_ub([physical_hidden], "float32")
            abs_ub = T.alloc_ub([physical_hidden], "float32")
            scale_ub = T.alloc_ub([1], "float32")
            input_fp16_ub = T.alloc_ub([physical_hidden], "float16")
            payload_ub = T.alloc_ub([physical_hidden], "int8")

            with T.Scope("V"):
                if row < local_tokens:
                    T.copy(input[row, 0:hidden_size], input_ub, pad_value=0)
                    T.tile.cast(
                        input_fp32_ub,
                        input_ub,
                        mode="CAST_NONE",
                        count=physical_hidden,
                    )
                    T.tile.abs(abs_ub, input_fp32_ub)
                    T.reduce_max(abs_ub, scale_ub, dim=-1)
                    if scale_ub[0] == 0.0:
                        scale_ub[0] = 1.0
                    else:
                        scale_ub[0] = scale_ub[0] / 127.0
                    for column in T.Parallel(physical_hidden):
                        input_fp32_ub[column] = input_fp32_ub[column] / scale_ub[0]
                    T.tile.clamp(
                        input_fp32_ub,
                        input_fp32_ub,
                        -127.0,
                        127.0,
                        physical_hidden,
                    )
                    T.tile.round(input_fp32_ub, input_fp32_ub, physical_hidden)
                    T.tile.cast(
                        input_fp16_ub,
                        input_fp32_ub,
                        mode="CAST_NONE",
                        count=physical_hidden,
                    )
                    T.tile.cast(
                        payload_ub,
                        input_fp16_ub,
                        mode="CAST_NONE",
                        count=physical_hidden,
                    )
                    T.copy(payload_ub, payload[row, 0])
                    T.copy(scale_ub, scale[row : row + 1])

    return aclshmem_moe_quant_int8
