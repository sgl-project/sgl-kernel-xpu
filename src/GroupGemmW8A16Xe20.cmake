set(GROUP_GEMM_W8A16_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmW8A16Xe20LauncherInstance.cpp.in")
set(GROUP_GEMM_W8A16_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_w8a16_xe20")
set(GROUP_GEMM_W8A16_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_W8A16_XE20_GEN_DIR})

function(add_group_gemm_w8a16_xe20_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE WEIGHT_KINDS)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    foreach(weight_kind IN LISTS WEIGHT_KINDS)
        if(weight_kind STREQUAL "int8")
            set(IS_INT8 true)
        else()
            set(IS_INT8 false)
        endif()
        set(GEN_SRC
            "${GROUP_GEMM_W8A16_XE20_GEN_DIR}/GroupGemmW8A16Xe20_inst_${TILE_M}_${TILE_N}_${TILE_K}_${weight_kind}.cpp")
        configure_file(${GROUP_GEMM_W8A16_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND GROUP_GEMM_W8A16_XE20_INST_SRCS ${GEN_SRC})
    endforeach()
    set(GROUP_GEMM_W8A16_XE20_INST_SRCS ${GROUP_GEMM_W8A16_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

# Unified W8A16 tile menu: all scale layouts (scalar, gate+up, per-channel,
# 128x128 block, K-group, and 1x32 MXFP8) are dispatched at runtime inside one
# kernel per (tile, weight_kind).
add_group_gemm_w8a16_xe20_inst("_16" "_64" "_32" "_1, _4, _1" "_4, _1, _0" "fp8;int8")
add_group_gemm_w8a16_xe20_inst("_32" "_64" "_32" "_1, _4, _1" "_4, _1, _0" "fp8;int8")
add_group_gemm_w8a16_xe20_inst("_64" "_64" "_32" "_2, _4, _1" "_4, _1, _0" "fp8;int8")
add_group_gemm_w8a16_xe20_inst("_128" "_128" "_32" "_4, _4, _1" "_4, _1, _0" "fp8;int8")

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_W8A16_XE20_INST_SRCS})
