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

# Stage-1 debugging: build just ONE shape to cut build time while iterating on
# correctness. Configuration targeting the minimal test case:
#   num_tokens_per_expert=33, num_experts=8, fuse_act=False, with_bias=False,
#   activation_type=0 (silu), K*N=1024*1024 (small_weight branch)
# hits  <_128, _128, _32> / SG_4_2_1 / fuse_act=false / with_bias=false / act=0
# via the GroupGemmMxfp4Xe20.cpp dispatcher.
#
# Restore the full tile menu (commented out below) once correctness is verified.
add_group_gemm_mxfp4_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" 0 false false)

# Full menu for post-correctness restoration:
# foreach(act_type 0 1 2)
#     foreach(with_bias true false)
#         foreach(fuse_act true false)
#             add_group_gemm_mxfp4_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias})
#         endforeach()
#         add_group_gemm_mxfp4_xe20_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
#         add_group_gemm_mxfp4_xe20_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias})
#         add_group_gemm_mxfp4_xe20_inst("_256" "_64" "_32" "_8, _2, _1" "_2, _1, _0" ${act_type} true ${with_bias})
#         add_group_gemm_mxfp4_xe20_inst("_256" "_256" "_32" "_8, _4, _1" "_4, _1, _0" ${act_type} false ${with_bias})
#     endforeach()
# endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_MXFP4_XE20_INST_SRCS})
