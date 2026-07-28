set(GROUP_GEMM_MXFP4_W4A16_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmMxfp4W4A16Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_MXFP4_W4A16_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_mxfp4_w4a16_xe20")
set(GROUP_GEMM_MXFP4_W4A16_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_MXFP4_W4A16_XE20_GEN_DIR})

function(add_group_gemm_mxfp4_w4a16_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT WITH_BIAS)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    set(GEN_SRC
        "${GROUP_GEMM_MXFP4_W4A16_XE20_GEN_DIR}/GroupGemmMxfp4W4A16Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")

    configure_file(${GROUP_GEMM_MXFP4_W4A16_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_MXFP4_W4A16_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_MXFP4_W4A16_XE20_INST_SRCS ${GROUP_GEMM_MXFP4_W4A16_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# Tile menu (5 tile shapes × 3 activations × bias/fuse_act combos):
#   <_8,   _64,  _32> / SG_1_4_1   — avg_m ≤ 8, all fuse_act/bias combos
#   <_128, _64,  _32> / SG_4_2_1   — avg_m ≤ 128, fuse_act=true
#   <_128, _128, _32> / SG_4_2_1   — avg_m ≤ 128, fuse_act=false
#   <_256, _64,  _32> / SG_8_2_1   — avg_m > 128,  fuse_act=true
#   <_256, _256, _32> / SG_8_4_1   — avg_m > 128,  fuse_act=false

# Instantiation matrix: act_type ∈ {0 silu, 2 swiglu_gpt_oss, 4 swiglu_deepseek_v4}
# × with_bias ∈ {false, true} × the fuse_act tile menu above.
#
# act_type 2 (swiglu_gpt_oss) and with_bias=true were re-enabled here to serve
# gpt-oss-20b (GptOssForCausalLM), whose MoE uses the gpt-oss gated activation
# and per-channel mlp1/mlp2 biases. The original prune kept only DSV4's
# {0,4}+no-bias variants to bound Level Zero per-context module count under
# TP>1; at TP=1 the larger matrix fits. The dispatch-side TORCH_CHECKs in
# src/sycl/GroupGemmMxfp4W4A16Xe20.cpp are relaxed to match. If you reintroduce
# TP>1 and hit L0 module-pool exhaustion, narrow this matrix back down.
#   act_type 0 = silu, 2 = swiglu_gpt_oss (clamp + (up+1) gate), 4 = swiglu_deepseek_v4
foreach(act_type 0 2 4)
    foreach(with_bias false true)
        foreach(fuse_act true false)
            add_group_gemm_mxfp4_w4a16_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias})
        endforeach()

        add_group_gemm_mxfp4_w4a16_xe20_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
        add_group_gemm_mxfp4_w4a16_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias})
        add_group_gemm_mxfp4_w4a16_xe20_inst("_256" "_64" "_32" "_8, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
        add_group_gemm_mxfp4_w4a16_xe20_inst("_256" "_256" "_32" "_8, _4, _1" "_4, _1, _0" ${act_type} false ${with_bias})
    endforeach()
endforeach()

list(APPEND device_cpp_xe20 ${GROUP_GEMM_MXFP4_W4A16_XE20_INST_SRCS})
