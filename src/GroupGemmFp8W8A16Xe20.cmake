set(GROUP_GEMM_FP8_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmFp8W8A16Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_FP8_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_fp8_xe20")
set(GROUP_GEMM_FP8_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_FP8_XE20_GEN_DIR})

function(add_group_gemm_fp8_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE SCALE_MODES)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    foreach(scale_mode IN LISTS SCALE_MODES)
        if(scale_mode STREQUAL "block")
            set(WEIGHT_SCALE_BLOCKED true)
        else()
            set(WEIGHT_SCALE_BLOCKED false)
        endif()
        set(SCALE_GEN_SRC
            "${GROUP_GEMM_FP8_XE20_GEN_DIR}/GroupGemmFp8Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_${scale_mode}.cpp")
        configure_file(${GROUP_GEMM_FP8_XE20_TEMPLATE} ${SCALE_GEN_SRC} @ONLY)
        list(APPEND GROUP_GEMM_FP8_XE20_INST_SRCS ${SCALE_GEN_SRC})
    endforeach()
    set(GROUP_GEMM_FP8_XE20_INST_SRCS ${GROUP_GEMM_FP8_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# W8A16 scalar/block menu: 16x64x32 and 32x64x32 for small/medium M,
# 64x64x32 for medium-M long-K scalar GEMMs, plus 128x128x16 for
# large-M scalar GEMMs. Activation is external.
add_group_gemm_fp8_xe20_inst("_16" "_64" "_32" "_1, _4, _1" "_4, _1, _0" "scalar;block")
add_group_gemm_fp8_xe20_inst("_32" "_64" "_32" "_1, _4, _1" "_4, _1, _0" "scalar;block")
add_group_gemm_fp8_xe20_inst("_64" "_64" "_32" "_2, _4, _1" "_4, _1, _0" "scalar")
add_group_gemm_fp8_xe20_inst("_128" "_128" "_16" "_4, _2, _1" "_2, _1, _0" "scalar;block")

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_FP8_XE20_INST_SRCS})
