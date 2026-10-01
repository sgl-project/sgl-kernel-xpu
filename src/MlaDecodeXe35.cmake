# Generate MLA decode kernel instantiation files for the xe35 (CRI) arch bucket.
# Mirrors MlaDecodeXe20.cmake but targets the xe35 arch bucket: each TU includes
# the xe35 device stack and pins SYCL_INTEL_TARGET=35. These land in
# device_cpp_xe35 and are compiled only when DPCPP_SYCL_TARGET matches "cri".
#
# Both arch buckets instantiate the same (ELEM_TAG, PAGE_SIZE, Q_TILE) grid; the device
# code under sycl/kernels/mla/xe35 starts out identical to its xe20 twin and
# diverges only where Xe3P needs different tiles/atoms. The arch split exists so
# that divergence has somewhere to live -- and so CRI stops compiling the Xe2
# vISA that SYCL_INTEL_TARGET=20 selects inside cutlass-sycl.

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
            set(ARCH_TAG xe35)
            set(SYCL_TARGET 35)
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_decode_kernel_${ELEM_TAG}_${PAGE_SIZE}_q${Q_TILE}_${ARCH_TAG}.cpp")
            configure_file(${MLA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_xe35 ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
