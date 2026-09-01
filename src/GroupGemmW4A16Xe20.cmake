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

# Policy menu. select_w4a16_policy_id() in GroupGemmW4A16Xe20.cpp picks one per
# call from the average rows-per-expert and gemm_n. The list order must match the
# policy_id switch there and the w4a16_policy() table in src/jit/moe_jit.cpp.
#   id 0  w4a16_policy_m_8_n_64    <_8,  _64,  _32>  — avg_m <= 4
#   id 1  w4a16_policy_m_16_n_64   <_16, _64,  _32>  — avg_m <= 8
#   id 2  w4a16_policy_m_32_n_64   <_32, _64,  _32>  — small avg_m
#   id 3  w4a16_policy_m_64_n_128  <_64, _128, _32>  — mid avg_m
#   id 4  w4a16_policy_m_128_n_128 <_128,_128, _32>  — large avg_m
#   id 5  w4a16_policy_m_64_n_256  <_64, _256, _32>  — GPT-OSS prefill GEMM1
# id 5 is the tile a5bcd5c ("Optimize W4A16 MoE GEMM for GPT-OSS prefill")
# measured its win with; it is the only tile above avg_m = 32 with ATOM_M == 1,
# so it is the only one whose sub-groups do not re-dequantise the same B rows.
# The rows-per-expert fill model in #446 cannot express that term, so the policy
# is kept as an explicit sixth entry rather than folded into the score table.
# group_size (32/64/128/256) is compiled into every unit as a runtime branch,
# so it does not multiply the instance count. Total: 6 policies x 2 (int4/mxfp4)
# x 2 (bf16/fp16 activation) = 24 units.
foreach(policy w4a16_policy_m_8_n_64 w4a16_policy_m_16_n_64 w4a16_policy_m_32_n_64
               w4a16_policy_m_64_n_128 w4a16_policy_m_128_n_128 w4a16_policy_m_64_n_256)
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
