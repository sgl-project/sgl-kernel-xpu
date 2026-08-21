set(GROUP_GEMM_FP8_W8A8_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmFp8W8A8Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_fp8_w8a8_xe20")
set(GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR})

function(add_group_gemm_fp8_w8a8_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    foreach(with_bias false true)
        set(WITH_BIAS ${with_bias})
        set(GEN_SRC
            "${GROUP_GEMM_FP8_W8A8_XE20_GEN_DIR}/GroupGemmFp8W8A8Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")
        configure_file(${GROUP_GEMM_FP8_W8A8_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS ${GEN_SRC})
    endforeach()
    set(GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS ${GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# FP8 instantiation matrix - deliberately smaller than the bf16/MXFP4
# matrices on BOTH axes (see GroupGemmFp8W8A8Xe20.cpp header comment):
#   - tile menu: _8/_16/_32 x _64 (SG_1_4_1); the initial _128 variants were
#     measured slower than Tile32 for wider-N and larger-M shapes and are not
#     instantiated until a narrow-N workload justifies their binary cost.
#   - activation is intentionally outside the FP8 GEMMs: tested Xe2 TP2/4/8
#     shapes show no split-path regression, and this removes ActType variants.
#     The Python path uses shared XPU activation kernels between GEMM1 and GEMM2.
# Generate separate WithBias=false/true instances so the epilogue can eliminate
# the bias decision entirely while the host dispatches based on bias presence.
# Keep both GEMM shapes in the shared matrix; the runtime uses Tile32 for
# avg_m above 16, including the TP>1 local shapes measured on Xe2.
add_group_gemm_fp8_w8a8_xe20_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 false)
add_group_gemm_fp8_w8a8_xe20_inst("_16" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 false)
add_group_gemm_fp8_w8a8_xe20_inst("_32" "_64" "_32" "_1, _4, _1" "_4, _1, _0" 0 false)

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_FP8_W8A8_XE20_INST_SRCS})
