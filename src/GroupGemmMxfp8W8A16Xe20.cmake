set(GROUP_GEMM_MXFP8_W8A16_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmMxfp8W8A16Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_MXFP8_W8A16_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_mxfp8_w8a16_xe20")
set(GROUP_GEMM_MXFP8_W8A16_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_MXFP8_W8A16_XE20_GEN_DIR})

function(add_group_gemm_mxfp8_w8a16_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT WITH_BIAS)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    set(GEN_SRC
        "${GROUP_GEMM_MXFP8_W8A16_XE20_GEN_DIR}/GroupGemmMxfp8W8A16Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")

    configure_file(${GROUP_GEMM_MXFP8_W8A16_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_MXFP8_W8A16_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_MXFP8_W8A16_XE20_INST_SRCS ${GROUP_GEMM_MXFP8_W8A16_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# Dense MXFP8 W8A16 linear only needs the non-fused, silu(act=0) slice, with and
# without bias, across the three non-fused tile shapes keyed on avg_m (matching
# the mxfp4 non-fused tiles):
#   <_8,   _64,  _32> / SG_1_4_1   — avg_m ≤ 8
#   <_128, _128, _32> / SG_4_2_1   — avg_m ≤ 128 (small weight)
#   <_256, _256, _32> / SG_8_4_1   — avg_m > 128
# 3 tiles × {no-bias, bias} = 6 TUs. The dispatch-side declarations in
# src/sycl/GroupGemmMxfp8W8A16Xe20.cpp must match this matrix exactly.
foreach(with_bias false true)
    add_group_gemm_mxfp8_w8a16_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 false ${with_bias})
    add_group_gemm_mxfp8_w8a16_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" 0 false ${with_bias})
    add_group_gemm_mxfp8_w8a16_xe20_inst("_256" "_256" "_32" "_8, _4, _1" "_4, _1, _0" 0 false ${with_bias})
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_MXFP8_W8A16_XE20_INST_SRCS})
