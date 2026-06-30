# GDN (Gated DeltaNet) attention Xe2 kernels.
# The host interface + recurrent path + conv1d kernels live in
# sycl/gdn_attention_Xe20.cpp (auto-globbed). The two Xe2 wrapper TUs below
# live in a subdir and are added to the Xe20 source list explicitly so they
# compile with -device bmg.
list(APPEND device_cpp_xe20
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/gdn_attn/xe_2/l2norm.cpp")
