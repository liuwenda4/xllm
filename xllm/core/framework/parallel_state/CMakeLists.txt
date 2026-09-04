include(cc_library)

set(XLLM_HAS_ACLSHMEM OFF CACHE INTERNAL
    "Whether the optional ACLSHMEM runtime is available" FORCE)
if(USE_NPU AND DEFINED ENV{ACLSHMEM_HOME} AND
   NOT "$ENV{ACLSHMEM_HOME}" STREQUAL "")
  find_path(ACLSHMEM_INCLUDE_DIR
    NAMES shmem.h
    HINTS "$ENV{ACLSHMEM_HOME}/include"
    NO_DEFAULT_PATH)
  find_library(ACLSHMEM_LIBRARY
    NAMES shmem
    HINTS
      "$ENV{ACLSHMEM_HOME}/lib"
      "$ENV{ACLSHMEM_HOME}/lib64"
      "$ENV{ACLSHMEM_HOME}/backends/910"
    NO_DEFAULT_PATH)
  if(ACLSHMEM_INCLUDE_DIR AND ACLSHMEM_LIBRARY)
    set(XLLM_HAS_ACLSHMEM ON CACHE INTERNAL
        "Whether the optional ACLSHMEM runtime is available" FORCE)
    message(STATUS "ACLSHMEM MoE resource enabled: ${ACLSHMEM_LIBRARY}")
  else()
    message(WARNING
      "ACLSHMEM_HOME is set but shmem.h/libshmem were not found; "
      "ACLSHMEM MoE remains unavailable")
  endif()
endif()

cc_library(
  NAME
    parallel_state
  HDRS
    mapping_npu.h
    npu_cp_plan.h
    npu_dp_ep_padding.h
    dit_mapping.h
    rank_generator.h
    parallel_args.h
    parallel_state.h
    process_group.h
    $<$<BOOL:${USE_NPU}>:npu_rank_table_env.h>
    $<$<BOOL:${USE_NPU}>:npu_process_group.h>
    $<$<BOOL:${USE_MLU}>:mlu_process_group.h>
    $<$<BOOL:${USE_MUSA}>:musa_process_group.h>
    $<$<OR:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_DCU}>>:cuda_process_group.h>
    $<$<BOOL:${USE_ILU}>:ilu_process_group.h>
    collective_communicator_base.h
    collective_communicator.h
    dit_collective_communicator.h
    $<$<BOOL:${USE_NPU}>:mega_moe_comm_resource.h>
    $<$<BOOL:${USE_NPU}>:aclshmem_moe_capability.h>
    $<$<BOOL:${XLLM_HAS_ACLSHMEM}>:shmem_comm_resource.h>
  SRCS
    mapping_npu.cpp
    npu_cp_plan.cpp
    npu_dp_ep_padding.cpp
    dit_mapping.cpp
    parallel_state.cpp
    parallel_state_async.cpp
    process_group.cpp
    $<$<BOOL:${USE_NPU}>:npu_rank_table_env.cpp>
    $<$<BOOL:${USE_NPU}>:npu_process_group.cpp>
    $<$<BOOL:${USE_NPU}>:mega_moe_comm_resource.cpp>
    $<$<BOOL:${USE_NPU}>:aclshmem_moe_capability.cpp>
    $<$<BOOL:${XLLM_HAS_ACLSHMEM}>:shmem_comm_resource.cpp>
    collective_communicator.cpp
    dit_collective_communicator.cpp
  DEPS
    :common
    torch
    $<$<BOOL:${USE_MLU}>:torch_mlu>
    $<$<BOOL:${USE_MUSA}>:torch_musa>
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:hccl>
    $<$<BOOL:${USE_NPU}>:hcomm>
    $<$<BOOL:${USE_NPU}>:xllm_atb_layers>
    gflags::gflags
    glog::glog
)

if(XLLM_HAS_ACLSHMEM)
  target_include_directories(parallel_state SYSTEM PUBLIC "${ACLSHMEM_INCLUDE_DIR}")
  target_link_libraries(parallel_state PUBLIC "${ACLSHMEM_LIBRARY}")
  target_compile_definitions(parallel_state PUBLIC XLLM_HAS_ACLSHMEM=1)
endif()
