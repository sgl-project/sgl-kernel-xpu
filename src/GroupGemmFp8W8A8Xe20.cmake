set(GROUP_GEMM_FP8_W8A8_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmFp8W8A8Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_fp8_w8a8_xe20")
set(GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR})

function(add_group_gemm_fp8_w8a8_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT WITH_BIAS)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    set(GEN_SRC
        "${GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR}/GroupGemmFp8W8A8Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")

    configure_file(${GROUP_GEMM_FP8_W8A8_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS ${GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# v1 instantiation matrix - deliberately smaller than the bf16/MXFP4
# matrices on BOTH axes (see GroupGemmFp8W8A8Xe20.cpp header comment):
#   - tile menu: _8/_16/_32 x _64 (SG_1_4_1) + a single _128 "large" tier
#     (SG_4_2_1), no _256 tiles yet - not a tuned decision, just a
#     conservative starting point given this mainloop decodes both A and B
#     from fp8 (more register pressure than bf16/MXFP4, which decode at
#     most one operand). Revisit once occupancy/spill data exists.
#   - act_type: only 0 (silu). GELU/SWIGLU_GPT_OSS/SWIGLU_DEEPSEEK_V4 are
#     all used by real model families that could plausibly ship fp8
#     checkpoints (grok/gemma4, gpt_oss, deepseek_v2/minimax_m3/step3p5
#     respectively) - which of those to add is a product decision on which
#     checkpoints are the fp8 validation targets, not inferred here.
foreach(with_bias false true)
    foreach(fuse_act true false)
        add_group_gemm_fp8_w8a8_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 ${fuse_act} ${with_bias})
        add_group_gemm_fp8_w8a8_xe20_inst("_16" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 ${fuse_act} ${with_bias})
        add_group_gemm_fp8_w8a8_xe20_inst("_32" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 ${fuse_act} ${with_bias})
    endforeach()

    add_group_gemm_fp8_w8a8_xe20_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" 0 true ${with_bias})
    add_group_gemm_fp8_w8a8_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" 0 false ${with_bias})
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS})
