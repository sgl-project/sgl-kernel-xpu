set(GROUP_GEMM_W4A16_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmW4A16Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_W4A16_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_w4a16_xe20")
set(GROUP_GEMM_W4A16_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_W4A16_XE20_GEN_DIR})

# Generate one translation unit per (policy, ElementS, ElementA) combo.
# For int4, ELEMENT_S matches ELEMENT_A. For mxfp4, both accepted tensor
# dtypes share the same E8M0 byte encoding, which the kernel reads as uint8_t.
function(add_group_gemm_w4a16_xe20_inst POLICY ELEMENT_S ELEMENT_A SANITIZED)
    set(GEN_SRC
        "${GROUP_GEMM_W4A16_XE20_GEN_DIR}/GroupGemmW4A16Xe20_inst_${POLICY}_${SANITIZED}.cpp")
    configure_file(${GROUP_GEMM_W4A16_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_W4A16_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_W4A16_XE20_INST_SRCS ${GROUP_GEMM_W4A16_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# fmha-cri production registry. The 64-row policies use single-M-subgroup
# layouts and have separate N-tail-skip instantiations, because that bit changes
# device code. select_w4a16_policy_id() chooses from this table per call.
# group_size (32/64/128/256) is compiled into every unit as a runtime branch,
# so it does not multiply the instance count. Total: 7 policies x 2
# (int4/mxfp4) x 2 (bf16/fp16 activation) = 28 units.
foreach(policy
        w4a16_launch_policy_m_8_n_64
        w4a16_launch_policy_m_16_n_64
        w4a16_launch_policy_m_32_n_64
        w4a16_launch_policy_m_64_n_128
        w4a16_launch_policy_m_64_n_128_skip
        w4a16_launch_policy_m_64_n_256
        w4a16_launch_policy_m_64_n_256_skip)
    foreach(act_tag bf16 fp16)
        if(act_tag STREQUAL "bf16")
            set(element_a "cutlass::bfloat16_t")
        else()
            set(element_a "cutlass::half_t")
        endif()
        # int4: scale and activation use the same dtype.
        add_group_gemm_w4a16_xe20_inst(${policy} "${element_a}" "${element_a}" "int4_${act_tag}")
        # mxfp4: read the raw E8M0 byte from uint8 or float8_e8m0fnu tensors.
        add_group_gemm_w4a16_xe20_inst(${policy} "uint8_t" "${element_a}" "mxfp4_${act_tag}")
    endforeach()
endforeach()

list(APPEND device_cpp_xe20 ${GROUP_GEMM_W4A16_XE20_INST_SRCS})
