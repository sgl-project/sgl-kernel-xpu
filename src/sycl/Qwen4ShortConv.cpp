#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

constexpr int WG_SIZE = 256;
constexpr int64_t CHANNELS_PER_WG = WG_SIZE;
constexpr int STATE_WG_SIZE = 64;
constexpr int64_t STATE_CHANNELS_PER_WG = STATE_WG_SIZE;
constexpr int STATE_VEC = 8;              // 16-byte global accesses when channels and pointers allow it
constexpr int64_t MAX_STATE_LEN = 16;     // matches the CUDA dispatcher; bounds the SLM tile
constexpr int64_t MAX_VERIFY_WIDTH = 32;  // matches the CUDA dispatcher; bounds the verify SLM tiles
constexpr int64_t PADDING_SLOT = 0;       // CUDA-graph padding slot; may repeat and is never advanced
constexpr int64_t MAX_GRID_DIM = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit

inline float round_bf16(float f) {
  return static_cast<float>(static_cast<bf16>(f));
}

// Same formula as the torch silu kernel, so the bf16 result matches eager bit for bit.
inline float silu(float x) {
  return x / (1.f + sycl::exp(-x));
}

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T v[N];
};

// conv_input[t] = [state[slot_t], x[t]] and state[slot_t] advances by one column, for one block of channels.
// The old state is staged in SLM before the barrier, so the in-place update never races with its own reads.
// Global state and conv_input blocks are contiguous, so they move in packs of VEC elements.
template <typename T, int VEC>
struct ShortConvStateKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using P = Pack<T, VEC>;
  T* state;
  const int64_t* state_indices;
  const T* x;
  T* conv_input;
  int64_t channels;
  int state_len;
  sycl::local_accessor<T, 1> old_state;
  sycl::local_accessor<T, 1> new_x;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    old_state = sycl::local_accessor<T, 1>(STATE_CHANNELS_PER_WG * state_len, cgh);
    new_x = sycl::local_accessor<T, 1>(STATE_CHANNELS_PER_WG, cgh);
  }

  void operator()(sycl::nd_item<2> item) const {
    const int64_t token = item.get_group(0);
    const int64_t first_channel = item.get_group(1) * STATE_CHANNELS_PER_WG;
    const int local_id = item.get_local_id(1);
    const int block_channels = sycl::min<int64_t>(STATE_CHANNELS_PER_WG, channels - first_channel);
    const int block_state = block_channels * state_len;
    const int out_len = state_len + 1;
    const int64_t slot = state_indices[token];
    P* slot_state = reinterpret_cast<P*>(state + (slot * channels + first_channel) * state_len);

    for (int i = local_id; i < block_state / VEC; i += STATE_WG_SIZE) {
      const P pack = slot_state[i];
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        old_state[i * VEC + k] = pack.v[k];
      }
    }
    for (int i = local_id; i < block_channels; i += STATE_WG_SIZE) {
      new_x[i] = x[token * channels + first_channel + i];
    }
    sycl::group_barrier(item.get_group());

    P* out = reinterpret_cast<P*>(conv_input + (token * channels + first_channel) * out_len);
    for (int i = local_id; i < block_channels * out_len / VEC; i += STATE_WG_SIZE) {
      P pack;
      int channel = i * VEC / out_len;
      int col = i * VEC - channel * out_len;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        pack.v[k] = col < state_len ? old_state[channel * state_len + col] : new_x[channel];
        if (++col == out_len) {
          col = 0;
          ++channel;
        }
      }
      out[i] = pack;
    }
    if (slot == PADDING_SLOT) {
      return;
    }
    for (int i = local_id; i < block_state / VEC; i += STATE_WG_SIZE) {
      P pack;
      int channel = i * VEC / state_len;
      int col = i * VEC - channel * state_len;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        pack.v[k] = col + 1 < state_len ? old_state[i * VEC + k + 1] : new_x[channel];
        if (++col == state_len) {
          col = 0;
          ++channel;
        }
      }
      slot_state[i] = pack;
    }
  }
};

// One work-group stages window[c] = [state[slot_r, c] as bf16, x[r * W : (r + 1) * W, c]] for one request and a
// block of channels in SLM, then writes intermediate[r, s] = window[:, s + 1 : s + 1 + L] (zeroed for invalid
// steps) and output[r * W + s, c] = residual + silu(depthwise conv of window[c]) (zero for invalid steps).
template <typename StateT, typename IdxT, typename InterT>
struct VerifyConvKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  const bf16* x;
  const bf16* residual;
  const bf16* weight;
  const StateT* state;
  const IdxT* state_indices;
  const bool* valid;
  bf16* output;
  InterT* intermediate;
  int64_t state_stride[3];
  int64_t index_stride;
  int64_t inter_stride[4];
  int64_t channels;
  int width;
  int state_len;
  int kernel_size;
  int dilation;
  sycl::local_accessor<bf16, 1> window;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    window = sycl::local_accessor<bf16, 1>(CHANNELS_PER_WG * (state_len + width), cgh);
  }

  void operator()(sycl::nd_item<2> item) const {
    const int64_t req = item.get_group(0);
    const int64_t first_channel = item.get_group(1) * CHANNELS_PER_WG;
    const int local_id = item.get_local_id(1);
    const int block_channels = sycl::min<int64_t>(CHANNELS_PER_WG, channels - first_channel);
    const int conv_len = state_len + width;
    const int64_t slot = static_cast<int64_t>(state_indices[req * index_stride]);

    const StateT* slot_state = state + slot * state_stride[0] + first_channel * state_stride[1];
    // (channel, col) of i advances by (stride_channels, stride_cols) per loop step, avoiding a division each time.
    const int stride_channels = state_len > 0 ? WG_SIZE / state_len : 0;
    const int stride_cols = WG_SIZE - stride_channels * state_len;
    const int first_channel_of_thread = state_len > 0 ? local_id / state_len : 0;
    const int first_col_of_thread = local_id - first_channel_of_thread * state_len;
    for (int i = local_id, channel = first_channel_of_thread, col = first_col_of_thread; i < block_channels * state_len;
         i += WG_SIZE) {
      window[channel * conv_len + col] =
          static_cast<bf16>(slot_state[channel * state_stride[1] + col * state_stride[2]]);
      channel += stride_channels;
      col += stride_cols;
      if (col >= state_len) {
        col -= state_len;
        ++channel;
      }
    }
    if (local_id < block_channels) {
      const bf16* req_x = x + req * width * channels + first_channel + local_id;
      for (int step = 0; step < width; ++step) {
        window[local_id * conv_len + state_len + step] = req_x[step * channels];
      }
    }
    sycl::group_barrier(item.get_group());

    if (local_id < block_channels) {
      // Taps are summed in fp32 in kernel order and rounded to bf16, matching the eager bf16 conv1d.
      const bf16* channel_weight = weight + (first_channel + local_id) * kernel_size;
      const int window_base = local_id * conv_len;
      for (int step = 0; step < width; ++step) {
        const int64_t token = req * width + step;
        const int64_t offset = token * channels + first_channel + local_id;
        if (!valid[token]) {
          output[offset] = bf16(0.f);
          continue;
        }
        float acc = 0.f;
        for (int k = 0; k < kernel_size; ++k) {
          acc += static_cast<float>(window[window_base + step + k * dilation]) * static_cast<float>(channel_weight[k]);
        }
        const float activated = round_bf16(silu(round_bf16(acc)));
        output[offset] = static_cast<bf16>(activated + static_cast<float>(residual[offset]));
      }
    }
    if (intermediate == nullptr) {
      return;
    }
    InterT* req_inter = intermediate + req * inter_stride[0] + first_channel * inter_stride[2];
    for (int step = 0; step < width; ++step) {
      const bool step_valid = valid[req * width + step];
      InterT* step_inter = req_inter + step * inter_stride[1];
      for (int i = local_id, channel = first_channel_of_thread, col = first_col_of_thread;
           i < block_channels * state_len;
           i += WG_SIZE) {
        const bf16 value = step_valid ? window[channel * conv_len + step + 1 + col] : bf16(0.f);
        step_inter[channel * inter_stride[2] + col * inter_stride[3]] = static_cast<InterT>(value);
        channel += stride_channels;
        col += stride_cols;
        if (col >= state_len) {
          col -= state_len;
          ++channel;
        }
      }
    }
  }
};

template <typename F>
void dispatch_bf16_or_float(at::ScalarType dtype, const char* name, F&& f) {
  if (dtype == at::kBFloat16) {
    f(bf16{});
  } else if (dtype == at::kFloat) {
    f(float{});
  } else {
    TORCH_CHECK(false, name, " must be bfloat16 or float32, got ", dtype);
  }
}

template <typename F>
void dispatch_index(at::ScalarType dtype, F&& f) {
  if (dtype == at::kInt) {
    f(int32_t{});
  } else if (dtype == at::kLong) {
    f(int64_t{});
  } else {
    TORCH_CHECK(false, "state_indices must be int32 or int64, got ", dtype);
  }
}

}  // namespace

namespace at::native::xpu {

// Decode short-conv state step: conv_input[N, C, L + 1] = [state[state_indices], x], then each non-padding
// slot of state[slots, C, L] advances in place to conv_input[:, :, 1:]. Real slots must be unique.
SGL_KERNEL_EXPORT void qwen4_short_conv_state(
    torch::Tensor& conv_input, torch::Tensor& state, torch::Tensor& state_indices, torch::Tensor& x) {
  CHECK_INPUT(conv_input);
  CHECK_INPUT(state);
  CHECK_INPUT(state_indices);
  CHECK_INPUT(x);
  TORCH_CHECK(
      state.scalar_type() == at::kBFloat16 || state.scalar_type() == at::kHalf,
      "state must be bfloat16 or float16, got ",
      state.scalar_type());
  TORCH_CHECK(x.scalar_type() == state.scalar_type(), "x must have the same dtype as state");
  TORCH_CHECK(conv_input.scalar_type() == state.scalar_type(), "conv_input must have the same dtype as state");
  TORCH_CHECK(state_indices.scalar_type() == at::kLong, "state_indices must be int64");
  TORCH_CHECK(state.dim() == 3, "state must have shape [slots, C, L]");
  const int64_t channels = state.size(1);
  const int64_t state_len = state.size(2);
  TORCH_CHECK(state_len > 0 && state_len <= MAX_STATE_LEN, "state length must be in [1, ", MAX_STATE_LEN, "]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must be 1-D");
  const int64_t tokens = state_indices.size(0);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == tokens && x.size(1) == channels, "x must have shape [N, C]");
  TORCH_CHECK(
      conv_input.dim() == 3 && conv_input.size(0) == tokens && conv_input.size(1) == channels &&
          conv_input.size(2) == state_len + 1,
      "conv_input must have shape [N, C, L + 1]");
  TORCH_CHECK(
      state_indices.device() == state.device() && x.device() == state.device() && conv_input.device() == state.device(),
      "all tensors must be on the same device");
  at::assert_no_overlap(conv_input, state);
  at::assert_no_overlap(conv_input, x);
  at::assert_no_overlap(state, x);
  if (tokens == 0 || channels == 0) {
    return;
  }
  const int64_t channel_blocks = (channels + STATE_CHANNELS_PER_WG - 1) / STATE_CHANNELS_PER_WG;
  TORCH_CHECK(
      tokens <= MAX_GRID_DIM && channel_blocks * STATE_WG_SIZE <= MAX_GRID_DIM, "qwen4_short_conv_state: too large");
  constexpr int64_t pack_bytes = STATE_VEC * 2;
  const bool vectorize = channels % STATE_VEC == 0 && reinterpret_cast<uintptr_t>(state.data_ptr()) % pack_bytes == 0 &&
                         reinterpret_cast<uintptr_t>(conv_input.data_ptr()) % pack_bytes == 0;

  c10::DeviceGuard guard(state.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  auto launch = [&](auto tag, auto vec) {
    using T = decltype(tag);
    ShortConvStateKernel<T, decltype(vec)::value> kernel{
        {},
        reinterpret_cast<T*>(state.data_ptr()),
        state_indices.data_ptr<int64_t>(),
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<T*>(conv_input.data_ptr()),
        channels,
        static_cast<int>(state_len),
        {},
        {}};
    sycl_kernel_submit(
        sycl::range<2>(tokens, channel_blocks * STATE_WG_SIZE), sycl::range<2>(1, STATE_WG_SIZE), q, kernel);
  };
  auto launch_dtype = [&](auto vec) {
    if (state.scalar_type() == at::kBFloat16) {
      launch(bf16{}, vec);
    } else {
      launch(fp16{}, vec);
    }
  };
  if (vectorize) {
    launch_dtype(std::integral_constant<int, STATE_VEC>{});
  } else {
    launch_dtype(std::integral_constant<int, 1>{});
  }
}

// Verify-step short conv: with conv_input[r] = [state[state_indices[r]] as bf16, x viewed as [R, W, C] transposed],
// output[R * W, C] = valid ? residual + silu(conv1d(conv_input, weight, dilation, groups=C)) : 0. If given,
// intermediate[r, s] (shape [>=R, >=W, C, L]) receives conv_input[r, :, s + 1 : s + 1 + L], zeroed where
// valid[r * W + s] is false. state is not modified.
SGL_KERNEL_EXPORT void qwen4_verify_conv(
    torch::Tensor& output,
    const std::optional<torch::Tensor>& intermediate,
    torch::Tensor& x,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& state,
    torch::Tensor& state_indices,
    torch::Tensor& valid,
    int64_t width,
    int64_t dilation) {
  CHECK_INPUT(output);
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  CHECK_INPUT(valid);
  CHECK_DEVICE(state);
  CHECK_DEVICE(state_indices);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be a 2-D bfloat16 tensor");
  TORCH_CHECK(state.dim() == 3 && state.size(1) == x.size(1), "state must have shape [slots, C, L]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must be 1-D");
  const int64_t requests = state_indices.size(0);
  const int64_t channels = x.size(1);
  const int64_t state_len = state.size(2);
  TORCH_CHECK(width > 0 && width <= MAX_VERIFY_WIDTH, "width must be in [1, ", MAX_VERIFY_WIDTH, "]");
  TORCH_CHECK(state_len <= MAX_STATE_LEN, "state length must be at most ", MAX_STATE_LEN);
  TORCH_CHECK(x.size(0) == requests * width, "x must have shape [R * W, C]");
  TORCH_CHECK(
      valid.scalar_type() == at::kBool && valid.dim() == 1 && valid.size(0) == x.size(0),
      "valid must be a bool tensor of shape [R * W]");
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16 && residual.sizes() == x.sizes(),
      "residual must be bfloat16 with shape [R * W, C]");
  TORCH_CHECK(
      output.scalar_type() == at::kBFloat16 && output.sizes() == x.sizes(),
      "output must be bfloat16 with shape [R * W, C]");
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16 && weight.dim() == 3 && weight.size(0) == channels && weight.size(1) == 1,
      "weight must be bfloat16 with shape [C, 1, K]");
  const int64_t kernel_size = weight.size(2);
  TORCH_CHECK(
      dilation > 0 && kernel_size > 0 && (kernel_size - 1) * dilation == state_len,
      "(kernel_size - 1) * dilation must equal state length ",
      state_len,
      ", got kernel_size=",
      kernel_size,
      ", dilation=",
      dilation);
  const bool has_intermediate = intermediate.has_value() && state_len > 0;
  if (intermediate.has_value()) {
    const auto& inter = *intermediate;
    CHECK_DEVICE(inter);
    TORCH_CHECK(
        inter.dim() == 4 && inter.size(0) >= requests && inter.size(1) >= width && inter.size(2) == channels &&
            inter.size(3) == state_len,
        "intermediate must have shape [>=R, >=W, C, L]");
    TORCH_CHECK(
        inter.scalar_type() == at::kBFloat16 || inter.scalar_type() == at::kFloat,
        "intermediate must be bfloat16 or float32, got ",
        inter.scalar_type());
    at::assert_no_internal_overlap(inter);
    for (const torch::Tensor* input : {&output, &x, &residual, &weight, &state}) {
      at::assert_no_overlap(inter, *input);
    }
  }
  // output may alias residual exactly: each element is read before it is written by the same thread.
  at::assert_no_partial_overlap(output, residual);
  for (const torch::Tensor* input : {&x, &weight, &state}) {
    at::assert_no_overlap(output, *input);
  }
  if (requests == 0 || channels == 0) {
    return;
  }
  const int64_t channel_blocks = (channels + CHANNELS_PER_WG - 1) / CHANNELS_PER_WG;
  TORCH_CHECK(requests <= MAX_GRID_DIM && channel_blocks * WG_SIZE <= MAX_GRID_DIM, "qwen4_verify_conv: too large");

  auto q = c10::xpu::getCurrentXPUStream().queue();
  const auto inter_dtype = has_intermediate ? intermediate->scalar_type() : at::kBFloat16;
  dispatch_bf16_or_float(state.scalar_type(), "state", [&](auto state_tag) {
    using StateT = decltype(state_tag);
    dispatch_index(state_indices.scalar_type(), [&](auto index_tag) {
      using IdxT = decltype(index_tag);
      dispatch_bf16_or_float(inter_dtype, "intermediate", [&](auto inter_tag) {
        using InterT = decltype(inter_tag);
        VerifyConvKernel<StateT, IdxT, InterT> kernel{
            {},
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<const bf16*>(residual.data_ptr()),
            reinterpret_cast<const bf16*>(weight.data_ptr()),
            reinterpret_cast<const StateT*>(state.data_ptr()),
            reinterpret_cast<const IdxT*>(state_indices.data_ptr()),
            valid.data_ptr<bool>(),
            reinterpret_cast<bf16*>(output.data_ptr()),
            has_intermediate ? reinterpret_cast<InterT*>(intermediate->data_ptr()) : nullptr,
            {state.stride(0), state.stride(1), state.stride(2)},
            state_indices.stride(0),
            {0, 0, 0, 0},
            channels,
            static_cast<int>(width),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(dilation),
            {}};
        if (has_intermediate) {
          for (int d = 0; d < 4; ++d) {
            kernel.inter_stride[d] = intermediate->stride(d);
          }
        }
        sycl_kernel_submit(sycl::range<2>(requests, channel_blocks * WG_SIZE), sycl::range<2>(1, WG_SIZE), q, kernel);
      });
    });
  });
}

}  // namespace at::native::xpu
