set(GROUP_GEMM_MXFP4_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmMxfp4Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_MXFP4_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_mxfp4_xe20")
set(GROUP_GEMM_MXFP4_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_MXFP4_XE20_GEN_DIR})

function(add_group_gemm_mxfp4_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT WITH_BIAS)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    set(GEN_SRC
        "${GROUP_GEMM_MXFP4_XE20_GEN_DIR}/GroupGemmMxfp4Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")

    configure_file(${GROUP_GEMM_MXFP4_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_MXFP4_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_MXFP4_XE20_INST_SRCS ${GROUP_GEMM_MXFP4_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# Full Stage-1 tile menu (4 tile shapes × 3 activations × 2 fuse_act × 2 bias
# = 60 launcher instantiations):
#   <_8,   _64,  _32> / SG_1_4_1   — avg_m ≤ 8 (decode-like)
#   <_128, _64,  _32> / SG_4_2_1   — avg_m ≤ 128, fuse_act=true
#   <_128, _128, _32> / SG_4_2_1   — avg_m ≤ 128, fuse_act=false
#   <_256, _64,  _32> / SG_8_2_1   — avg_m > 128,  fuse_act=true
#   <_256, _256, _32> / SG_8_4_1   — avg_m > 128,  fuse_act=false

foreach(act_type 0 1 2)
    foreach(with_bias true false)
        foreach(fuse_act true false)
            add_group_gemm_mxfp4_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias})
        endforeach()

        add_group_gemm_mxfp4_xe20_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
        add_group_gemm_mxfp4_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias})
        add_group_gemm_mxfp4_xe20_inst("_256" "_64" "_32" "_8, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
        add_group_gemm_mxfp4_xe20_inst("_256" "_256" "_32" "_8, _4, _1" "_4, _1, _0" ${act_type} false ${with_bias})
    endforeach()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_MXFP4_XE20_INST_SRCS})
