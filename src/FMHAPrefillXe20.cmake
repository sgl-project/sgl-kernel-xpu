# Generate FMHA prefill kernel instantiation files.
# Each HEAD_DIM is compiled as a separate translation unit to parallelize
# and speed up compilation.
#
# Tile shape mapping (HEAD_DIM -> TILED_Q, TILED_KV, NUM_SG):
#   64  -> 128, 64, 8
#   96  -> 128, 64, 8
#   128 -> 256, 32, 16
#   192 -> 256, 64, 32
#   256 -> 256, 64, 32
#   512 -> 256, 64, 32

set(FMHA_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_prefill_kernel.cpp.in")

# Define the per-HEAD_DIM tile configurations
# Format: HEAD_DIM;TILED_Q;TILED_KV;NUM_SG
set(FMHA_PREFILL_CONFIGS
    "64;128;64;8"
    "96;128;64;8"
    "128;256;32;16"
    "192;256;64;32"
    "256;256;64;32"
    "512;256;64;32"
)

foreach(CONFIG ${FMHA_PREFILL_CONFIGS})
    list(GET CONFIG 0 HEAD_DIM)
    list(GET CONFIG 1 TILED_Q)
    list(GET CONFIG 2 TILED_KV)
    list(GET CONFIG 3 NUM_SG)

    set(GENERATED_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_kernel_${HEAD_DIM}.cpp")
    configure_file(${FMHA_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_FILE})
endforeach()
