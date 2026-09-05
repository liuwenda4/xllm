# Copyright 2025-2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

import os

import tilelang

from xllm.python.kernels_npu.tilelang import aclshmem_moe_dispatch_int8 as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.aclshmem_moe_dispatch_int8 import (
    build_aclshmem_moe_dispatch_int8_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)

ACLSHMEM_ASCEND_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
}


class AclShmemMoeDispatchInt8Kernel(TilelangKernel):
    KERNEL_NAME = "aclshmem_moe_dispatch_int8"
    DISPATCH_SCHEMA = [
        DispatchField("local_tokens", "int32"),
        DispatchField("hidden_size", "int32"),
        DispatchField("topk", "int32"),
        DispatchField("ep_world_size", "int32"),
        DispatchField("local_experts", "int32"),
        DispatchField("rank", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"t4_h37_k2_ep2_le2_r{rank}",
            "local_tokens": 4,
            "hidden_size": 37,
            "topk": 2,
            "ep_world_size": 2,
            "local_experts": 2,
            "rank": rank,
        }
        for rank in range(2)
    ] + [
        {
            "variant_key": f"t4_h4096_k6_ep16_le16_r{rank}",
            "local_tokens": 4,
            "hidden_size": 4096,
            "topk": 6,
            "ep_world_size": 16,
            "local_experts": 16,
            "rank": rank,
        }
        for rank in range(16)
    ]

    @staticmethod
    def generate_source(
        local_tokens: int,
        hidden_size: int,
        topk: int,
        ep_world_size: int,
        local_experts: int,
        rank: int,
    ) -> str:
        tilelang.disable_cache()
        prim_func = build_aclshmem_moe_dispatch_int8_kernel(
            local_tokens=local_tokens,
            hidden_size=hidden_size,
            topk=topk,
            ep_world_size=ep_world_size,
            local_experts=local_experts,
            rank=rank,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=ACLSHMEM_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(prim_func)
        return kernel.kernel_source


if os.environ.get("XLLM_ENABLE_ACLSHMEM_MOE_AOT", "0") == "1":
    AclShmemMoeDispatchInt8Kernel = register_kernel(AclShmemMoeDispatchInt8Kernel)
