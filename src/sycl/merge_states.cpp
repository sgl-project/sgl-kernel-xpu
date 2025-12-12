#include <sycl/sycl.hpp>
#include <limits>
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include "Utils.h"

inline float to_float(float u) {
	    return u;
}

inline float to_float(sycl::half u) {
	    return static_cast<float>(u);
}

inline float to_float(sycl::ext::oneapi::bfloat16 u) {
	    return static_cast<float>(u); // SYCL BF16 -> float conversion built-in
}

inline void from_float(float &d, float s) {
	    d = s;
}

inline void from_float(sycl::half &d, float s) {
	    d = static_cast<sycl::half>(s);
}

inline void from_float(sycl::ext::oneapi::bfloat16 &d, float s) {
	    d = static_cast<sycl::ext::oneapi::bfloat16>(s);
}

template <typename scalar_t, typename pack_128b_t>
struct MergePrefixSuffix {
    const uint64_t total_threads;
    const uint32_t threads_per_head;
    const uint32_t num_heads;
    const uint32_t head_size;
    const uint32_t pack_size;
    const scalar_t* prefix_output;
    const scalar_t* suffix_output;
    scalar_t* output;
    const float* prefix_lse;
    const float* suffix_lse;
    float* output_lse;

    MergePrefixSuffix(
        uint64_t total_threads,
        uint32_t threads_per_head,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t pack_size,
        const scalar_t* prefix_output,
        const scalar_t* suffix_output,
        scalar_t* output,
        const float* prefix_lse,
        const float* suffix_lse,
        float* output_lse)
        : total_threads(total_threads),
          threads_per_head(threads_per_head),
          num_heads(num_heads),
          head_size(head_size),
          pack_size(pack_size),
          prefix_output(prefix_output),
          suffix_output(suffix_output),
          output(output),
          prefix_lse(prefix_lse),
          suffix_lse(suffix_lse),
          output_lse(output_lse) {}

    void operator()(sycl::nd_item<1> it) const {
        const uint64_t global_idx = it.get_global_linear_id();
        if (global_idx >= total_threads)
            return;

        const uint32_t token_head_idx = global_idx / threads_per_head;
        const uint32_t pack_idx       = global_idx % threads_per_head;

        const uint32_t token_idx = token_head_idx / num_heads;
        const uint32_t head_idx  = token_head_idx % num_heads;

        const uint32_t pack_offset = pack_idx * pack_size;
        const uint32_t head_offset =
            token_idx * num_heads * head_size +
            head_idx * head_size;

        const scalar_t* prefix_head_ptr = prefix_output + head_offset;
        const scalar_t* suffix_head_ptr = suffix_output + head_offset;
        scalar_t* output_head_ptr = output + head_offset;

        float p_lse = prefix_lse[token_idx * num_heads + head_idx];
        float s_lse = suffix_lse[token_idx * num_heads + head_idx];

        p_lse = sycl::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
        s_lse = sycl::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

        const float max_lse = sycl::fmax(p_lse, s_lse);
        p_lse -= max_lse;
        s_lse -= max_lse;

        const float p_se = sycl::exp(p_lse);
        const float s_se = sycl::exp(s_lse);
        const float out_se = p_se + s_se;
        const float p_scale = p_se / out_se;
        const float s_scale = s_se / out_se;

        if (pack_offset < head_size) {
            auto p_pack_ptr = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr);
            auto s_pack_ptr = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr);
            auto o_pack_ptr = reinterpret_cast<pack_128b_t*>(output_head_ptr);

            pack_128b_t p_out_pack = p_pack_ptr[pack_offset / pack_size];
            pack_128b_t s_out_pack = s_pack_ptr[pack_offset / pack_size];
            pack_128b_t o_out_pack;

            scalar_t* p_vals = reinterpret_cast<scalar_t*>(&p_out_pack);
            scalar_t* s_vals = reinterpret_cast<scalar_t*>(&s_out_pack);
            scalar_t* o_vals = reinterpret_cast<scalar_t*>(&o_out_pack);

            #pragma unroll
            for (uint32_t i = 0; i < pack_size; ++i) {
                const float pf = to_float(p_vals[i]);
                const float sf = to_float(s_vals[i]);
                const float of = pf * p_scale + sf * s_scale;
                from_float(o_vals[i], of);
            }

            o_pack_ptr[pack_offset / pack_size] = o_out_pack;
        }

        if (output_lse != nullptr && pack_idx == 0) {
            float out_lse = sycl::log(out_se) + max_lse;
            // The output_lse array is indexed by (token_idx * num_heads + head_idx)
            // It is assumed to be safe to write to this global memory location by only thread pack_idx == 0 
            output_lse[token_idx * num_heads + head_idx] = out_lse;
        }
    }
};
template <typename scalar_t>
void merge_attn_states_sycl(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t head_size)
{
    using pack_128b_t = sycl::vec<uint32_t,4>;

    const uint32_t pack_size = 16 / sizeof(scalar_t);
    const uint32_t threads_per_head = head_size / pack_size;
    const uint64_t total_threads =
        uint64_t(num_tokens) * num_heads * threads_per_head;

    const uint32_t local_size = 128;
    const uint64_t global_size =
        ((total_threads + local_size - 1) / local_size) * local_size;
    auto stream = at::xpu::getCurrentXPUStream();
    auto q = stream.queue();
    q.submit([&](sycl::handler &h) {
        MergePrefixSuffix<scalar_t, pack_128b_t> kernel_functor(
                total_threads,
                threads_per_head,
                num_heads,
                head_size,
                pack_size,
                prefix_output,
                suffix_output,
                output,
                prefix_lse,
                suffix_lse,
                output_lse
            );
        h.parallel_for(
        sycl::nd_range<1>(global_size, local_size),
        kernel_functor
    );
    });
}

#define SYCL_DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                          \
  {                                                                              \
    if (scalar_dtype == at::ScalarType::Float) {                                 \
      fn(float);                                                                 \
    } else if (scalar_dtype == at::ScalarType::Half) {                           \
      fn(sycl::half);                                                            \
    } else if (scalar_dtype == at::ScalarType::BFloat16) {                       \
      fn(sycl::ext::oneapi::bfloat16);                                           \
    } else {                                                                     \
      TORCH_CHECK(false, "Unsupported dtype for SYCL merge_state_v2_sycl: ",     \
                  scalar_dtype);                                                 \
    }                                                                            \
  }

template <typename scalar_t>
void merge_attn_states_launcher_sycl(
    const at::Tensor& prefix_output,
    const at::Tensor& prefix_lse,
    const at::Tensor& suffix_output,
    const at::Tensor& suffix_lse,
    at::Tensor& output,
    at::Tensor& output_lse)
{
    const uint32_t num_tokens = output.size(0);
    const uint32_t num_heads  = output.size(1);
    const uint32_t head_size  = output.size(2);

    const uint32_t pack_size = 16 / sizeof(scalar_t);
    TORCH_CHECK(head_size % pack_size == 0,
                "head_size must be multiple of pack_size:", pack_size);
    merge_attn_states_sycl<scalar_t>(                                                                                                                   \
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        reinterpret_cast<float*>(output_lse.data_ptr()),
        reinterpret_cast<const scalar_t*>(prefix_output.data_ptr()),
        reinterpret_cast<const float*>(prefix_lse.data_ptr()),
        reinterpret_cast<const scalar_t*>(suffix_output.data_ptr()),
        reinterpret_cast<const float*>(suffix_lse.data_ptr()),
        num_tokens, num_heads, head_size);
}



#define CALL_MERGE_ATTN_STATES_LAUNCHER_SYCL(scalar_t) \
  { merge_attn_states_launcher_sycl<scalar_t>(v_a, s_a, v_b, s_b, v_merged, s_merged); }


  void merge_state_v2(
    at::Tensor v_a,
    at::Tensor s_a,
    at::Tensor v_b,
    at::Tensor s_b,
    at::Tensor v_merged,
    at::Tensor s_merged)
{
    CHECK_INPUT(v_a);
    CHECK_INPUT(s_a);
    CHECK_INPUT(v_b);
    CHECK_INPUT(s_b);
    CHECK_INPUT(v_merged);
    CHECK_INPUT(s_merged);

    CHECK_DIM(3, v_a);
    CHECK_DIM(2, s_a);
    CHECK_DIM(3, v_b);
    CHECK_DIM(2, s_b);

    CHECK_SAME_SHAPE(v_a, v_b);
    CHECK_SAME_SHAPE(s_a, s_b);
    CHECK_EQ(v_a.size(0), s_a.size(0));
    CHECK_EQ(v_a.size(1), s_b.size(1));

    SYCL_DISPATCH_BY_SCALAR_DTYPE(
        v_merged.dtype(),
        CALL_MERGE_ATTN_STATES_LAUNCHER_SYCL
    );
}
