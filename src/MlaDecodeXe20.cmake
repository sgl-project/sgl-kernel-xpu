# Generate MLA decode kernel instantiation files for the xe20 (BMG) arch bucket.
# Each (ELEM_TAG, PAGE_SIZE, Q_TILE) combination is compiled as a separate
# library to parallelize and speed up compilation.
#
# Each TU includes the xe20 device stack and pins SYCL_INTEL_TARGET=20. These land
# in device_cpp_xe20 and are compiled only when DPCPP_SYCL_TARGET matches "bmg";
# MlaDecodeXe35.cmake is the xe35/CRI twin. ARCH_TAG is part of the generated
# filename so the two generators cannot collide on an output path (or on the
# derived sgl-ops-sycl-<name> library target).

set(MLA_DECODE_ELEM_TAGS half bf16)
set(MLA_DECODE_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")
set(MLA_DECODE_PAGE_SIZES 16 32 64 128)
# Query tokens per request (seq_len_q). 1 is single-token decode; 2/4/8/16 are
# the multi-token (MTP / speculative) decode variants with a causal mask among
# the new tokens. Must match the Q_TILE ladder in mla_decode.cpp and the
# declarations in kernels/mla/<arch>/device/mla_decode_dispatch.hpp.
set(MLA_DECODE_Q_TILES 1 2 4 8 16)

set(MLA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_decode_kernel.cpp.in")

list(LENGTH MLA_DECODE_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET MLA_DECODE_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_DECODE_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    foreach(PAGE_SIZE ${MLA_DECODE_PAGE_SIZES})
        foreach(Q_TILE ${MLA_DECODE_Q_TILES})
            set(ARCH_TAG xe20)
            set(SYCL_TARGET 20)
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_decode_kernel_${ELEM_TAG}_${PAGE_SIZE}_q${Q_TILE}_${ARCH_TAG}.cpp")
            configure_file(${MLA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
