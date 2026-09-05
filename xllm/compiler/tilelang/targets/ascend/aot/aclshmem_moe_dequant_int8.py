# Copyright 2025-2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

import os

import tilelang

from xllm.python.kernels_npu.tilelang import aclshmem_moe_dequant_int8 as kernel_impl
from xllm.python.kernels_npu.tilelang.aclshmem_moe_dequant_int8 import (
    build_aclshmem_moe_dequant_int8_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl,)

ACLSHMEM_ASCEND_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
}


class AclShmemMoeDequantInt8Kernel(TilelangKernel):
    KERNEL_NAME = "aclshmem_moe_dequant_int8"
    DISPATCH_SCHEMA = [
        DispatchField("max_capacity", "int32"),
        DispatchField("hidden_size", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "c16_h37",
            "max_capacity": 16,
            "hidden_size": 37,
        },
        {
            "variant_key": "c1024_h4096",
            "max_capacity": 1024,
            "hidden_size": 4096,
        },
    ]

    @staticmethod
    def generate_source(max_capacity: int, hidden_size: int) -> str:
        tilelang.disable_cache()
        prim_func = build_aclshmem_moe_dequant_int8_kernel(
            max_capacity=max_capacity,
            hidden_size=hidden_size,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=ACLSHMEM_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(prim_func)
        return kernel.kernel_source


if os.environ.get("XLLM_ENABLE_ACLSHMEM_MOE_AOT", "0") == "1":
    AclShmemMoeDequantInt8Kernel = register_kernel(AclShmemMoeDequantInt8Kernel)
