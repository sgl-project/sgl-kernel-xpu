set(GROUP_GEMM_SIMD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmSIMDLauncherInstance.cpp.in")
set(GROUP_GEMM_SIMD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_simd")
set(GROUP_GEMM_SIMD_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_SIMD_GEN_DIR})

function(add_group_gemm_simd_inst TILE_M TILE_N TILE_K SG_SHAPE SG_STRIDE ACT_TYPE FUSE_ACT WITH_BIAS SYCL_INTEL_TARGET)
    set(TILE "Shape<${TILE_M}, ${TILE_N}, ${TILE_K}>")
    set(SGLAYOUT "Layout<Shape<${SG_SHAPE}>, Stride<${SG_STRIDE}>>")
    set(GEN_SRC
        "${GROUP_GEMM_SIMD_GEN_DIR}/GroupGemmSIMD_inst_xe${SYCL_INTEL_TARGET}_${TILE_M}_${TILE_N}_${TILE_K}_a${ACT_TYPE}_f${FUSE_ACT}_b${WITH_BIAS}.cpp")

    configure_file(${GROUP_GEMM_SIMD_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_SIMD_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_SIMD_INST_SRCS ${GROUP_GEMM_SIMD_INST_SRCS} PARENT_SCOPE)
endfunction()

set(GROUP_GEMM_SIMD_XE20_SRCS)
set(GROUP_GEMM_SIMD_XE35_SRCS)

foreach(target 20 35)
    set(GROUP_GEMM_SIMD_INST_SRCS)
    foreach(act_type 0 1 2 3)
        if(act_type EQUAL 3)
            set(with_bias_list false)
        else()
            set(with_bias_list true false)
        endif()
        foreach(with_bias ${with_bias_list})
            # All activation types now support both fused and unfused paths
            set(fuse_act_list true false)

            foreach(fuse_act ${fuse_act_list})
                add_group_gemm_simd_inst("_8" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias} ${target})
                add_group_gemm_simd_inst("_16" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias} ${target})
                add_group_gemm_simd_inst("_32" "_64" "_32" "_1, _4, _1" "_4, _1, _0" ${act_type} ${fuse_act} ${with_bias} ${target})
            endforeach()

            # For larger tiles, only instantiate the specific fuse_act values that are actually used
            foreach(fuse_act ${fuse_act_list})
                if(fuse_act)
                    add_group_gemm_simd_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} ${fuse_act} ${with_bias} ${target})
                    add_group_gemm_simd_inst("_256" "_64" "_32" "_8, _2, _1" "_2, _1, _0" ${act_type} ${fuse_act} ${with_bias} ${target})
                endif()
            endforeach()

            # These are always false
            add_group_gemm_simd_inst("_128" "_64" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias} ${target})
            add_group_gemm_simd_inst("_128" "_128" "_32" "_4, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias} ${target})
            add_group_gemm_simd_inst("_256" "_64" "_32" "_8, _2, _1" "_2, _1, _0" ${act_type} false ${with_bias} ${target})
            add_group_gemm_simd_inst("_256" "_256" "_32" "_8, _4, _1" "_4, _1, _0" ${act_type} false ${with_bias} ${target})
        endforeach()
    endforeach()

    if(target EQUAL 20)
        set(GROUP_GEMM_SIMD_XE20_SRCS ${GROUP_GEMM_SIMD_INST_SRCS})
    elseif(target EQUAL 35)
        set(GROUP_GEMM_SIMD_XE35_SRCS ${GROUP_GEMM_SIMD_INST_SRCS})
    endif()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GROUP_GEMM_SIMD_XE20_SRCS})
list(APPEND ATen_XPU_SYCL_XE35 ${GROUP_GEMM_SIMD_XE35_SRCS})
