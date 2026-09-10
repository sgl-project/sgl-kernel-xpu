# GDN (Gated DeltaNet) attention Xe2 kernels.
# The host interface + recurrent path + conv1d kernels live in
# sycl/gdn_attention.cpp (auto-globbed). The two wrapper TUs below
# live in a subdir and are added to the source list explicitly so they
# compile with -device bmg.
#
# The GLOB_RECURSE in the parent CMakeLists already picked these files up and
# classified them as "common" (their parent dir is gdn_attn/, not xe20/), so
# strip them from the common bucket before appending to the Xe20 bucket to
# avoid creating duplicate sycl_add_library targets.
set(_gdn_attn_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/gdn_attn/chunk_gated_delta_rule.cpp")

list(REMOVE_ITEM device_cpp_common ${_gdn_attn_srcs})
list(APPEND device_cpp_xe20 ${_gdn_attn_srcs})
list(APPEND device_cpp_xe35 ${_gdn_attn_srcs})
unset(_gdn_attn_srcs)
