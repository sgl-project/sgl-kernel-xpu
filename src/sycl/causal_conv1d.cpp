/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Causal conv1d forward/update kernels for XPU (SYCL).
// Implementation aims to mirror CUDA-side behavior in
// sgl-kernel/csrc/mamba/causal_conv1d.cu.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>

#define CHECK_SHAPE(x, ...)                                                                  \
	TORCH_CHECK((x).sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

template <typename scalar_t, int kWidth>
struct CausalConv1dFwdKernel {
	scalar_t* __restrict__ x_ptr;
	scalar_t* __restrict__ w_ptr;
	scalar_t* __restrict__ bias_ptr;
	scalar_t* __restrict__ conv_states_ptr;
	const int* __restrict__ query_start_loc_ptr;
	const bool* __restrict__ has_initial_state_ptr;
	const int* __restrict__ cache_indices_ptr;
	scalar_t* __restrict__ out_ptr;

	int batch;
	int dim;
	int seqlen;
	int64_t pad_slot_id;
	bool silu_activation;
	bool varlen;

	int x_batch_stride;
	int x_c_stride;
	int x_l_stride;
	int w_c_stride;
	int w_w_stride;
	int cs_batch_stride;
	int cs_c_stride;
	int cs_l_stride;
	int out_batch_stride;
	int out_c_stride;
	int out_l_stride;

	int conv_state_len;

	void operator()(sycl::nd_item<2> item) const {
		const int batch_id = item.get_group(0);
		const int channel_id = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
		if (batch_id >= batch || channel_id >= dim) {
			return;
		}

		const int seq_start = varlen ? query_start_loc_ptr[batch_id] : 0;
		const int cur_seqlen = varlen ? (query_start_loc_ptr[batch_id + 1] - seq_start) : seqlen;
		if (cur_seqlen <= 0) {
			return;
		}

		const int cache_idx = (cache_indices_ptr != nullptr) ? cache_indices_ptr[batch_id] : batch_id;
		if (cache_idx == static_cast<int>(pad_slot_id)) {
			return;
		}
		const bool has_init = (has_initial_state_ptr != nullptr) ? has_initial_state_ptr[batch_id] : false;

		float wt[kWidth];
#pragma unroll
		for (int w = 0; w < kWidth; ++w) {
			wt[w] = static_cast<float>(w_ptr[channel_id * w_c_stride + w * w_w_stride]);
		}
		const float bias_val = (bias_ptr != nullptr) ? static_cast<float>(bias_ptr[channel_id]) : 0.f;

		float x_vals[kWidth];
#pragma unroll
		for (int w = 0; w < kWidth - 1; ++w) {
			if (conv_states_ptr != nullptr && has_init) {
				x_vals[w] = static_cast<float>(
						conv_states_ptr[cache_idx * cs_batch_stride + channel_id * cs_c_stride + w * cs_l_stride]);
			} else {
				x_vals[w] = 0.f;
			}
		}
		x_vals[kWidth - 1] = 0.f;

		for (int t = 0; t < cur_seqlen; ++t) {
			const int x_offset = varlen
															 ? ((seq_start + t) * x_batch_stride + channel_id * x_c_stride)
															 : (batch_id * x_batch_stride + channel_id * x_c_stride + t * x_l_stride);
			const float x_t = static_cast<float>(x_ptr[x_offset]);
			x_vals[kWidth - 1] = x_t;

			float out_val = bias_val;
#pragma unroll
			for (int w = 0; w < kWidth; ++w) {
				out_val += wt[w] * x_vals[w];
			}

			if (silu_activation) {
				out_val = out_val / (1.f + sycl::exp(-out_val));
			}

			const int out_offset = varlen
																 ? ((seq_start + t) * out_batch_stride + channel_id * out_c_stride)
																 : (batch_id * out_batch_stride + channel_id * out_c_stride + t * out_l_stride);
			out_ptr[out_offset] = static_cast<scalar_t>(out_val);

#pragma unroll
			for (int w = 0; w < kWidth - 1; ++w) {
				x_vals[w] = x_vals[w + 1];
			}
		}

		if (conv_states_ptr != nullptr) {
#pragma unroll
			for (int w = 0; w < kWidth - 1; ++w) {
				conv_states_ptr[cache_idx * cs_batch_stride + channel_id * cs_c_stride + w * cs_l_stride] =
						static_cast<scalar_t>(x_vals[w]);
			}
			for (int w = kWidth - 1; w < conv_state_len; ++w) {
				conv_states_ptr[cache_idx * cs_batch_stride + channel_id * cs_c_stride + w * cs_l_stride] =
						static_cast<scalar_t>(0.f);
			}
		}
	}
};

template <typename scalar_t, int kWidth>
struct CausalConv1dUpdateKernel {
	scalar_t* __restrict__ x_ptr;
	scalar_t* __restrict__ w_ptr;
	scalar_t* __restrict__ bias_ptr;
	scalar_t* __restrict__ conv_state_ptr;
	const int* __restrict__ conv_state_indices_ptr;
	const int* __restrict__ cache_seqlens_ptr;
	scalar_t* __restrict__ out_ptr;

	int batch;
	int dim;
	int seqlen;
	int conv_state_len;
	int64_t pad_slot_id;
	bool silu_activation;

	int x_batch_stride;
	int x_c_stride;
	int x_l_stride;
	int w_c_stride;
	int w_w_stride;
	int cs_batch_stride;
	int cs_c_stride;
	int cs_l_stride;
	int out_batch_stride;
	int out_c_stride;
	int out_l_stride;

	void operator()(sycl::nd_item<2> item) const {
		const int batch_id = item.get_group(0);
		const int channel_id = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
		if (batch_id >= batch || channel_id >= dim) {
			return;
		}

		const int cache_idx = (conv_state_indices_ptr != nullptr) ? conv_state_indices_ptr[batch_id] : batch_id;
		if (cache_idx == static_cast<int>(pad_slot_id)) {
			return;
		}
		const bool circular = (cache_seqlens_ptr != nullptr);

		float wt[kWidth];
#pragma unroll
		for (int w = 0; w < kWidth; ++w) {
			wt[w] = static_cast<float>(w_ptr[channel_id * w_c_stride + w * w_w_stride]);
		}
		const float bias_val = (bias_ptr != nullptr) ? static_cast<float>(bias_ptr[channel_id]) : 0.f;

		scalar_t* conv_state_channel =
				conv_state_ptr + cache_idx * cs_batch_stride + channel_id * cs_c_stride;
		const int state_len = conv_state_len;
		const int advance_len = seqlen;

		int cache_seqlen = circular ? (cache_seqlens_ptr[batch_id] % state_len) : 0;
		int update_idx = cache_seqlen - (kWidth - 1);
		update_idx = (update_idx < 0) ? (update_idx + state_len) : update_idx;

		float x_vals[kWidth] = {0.f};
		if (!circular) {
			const int shift_count = state_len - advance_len - (kWidth - 1);
			for (int i = 0; i < shift_count; ++i) {
				conv_state_channel[i * cs_l_stride] =
						conv_state_channel[(i + advance_len) * cs_l_stride];
			}

			for (int i = 0; i < kWidth - 1; ++i) {
				const scalar_t state_val = conv_state_channel[(state_len - (kWidth - 1) + i) * cs_l_stride];
				const int write_idx = state_len - advance_len - (kWidth - 1) + i;
				if (i < advance_len + (kWidth - 1) && write_idx >= 0) {
					conv_state_channel[write_idx * cs_l_stride] = state_val;
				}
				x_vals[i] = static_cast<float>(state_val);
			}
		} else {
			for (int i = 0; i < kWidth - 1; ++i) {
				const scalar_t state_val = conv_state_channel[update_idx * cs_l_stride];
				x_vals[i] = static_cast<float>(state_val);
				++update_idx;
				update_idx = (update_idx >= state_len) ? (update_idx - state_len) : update_idx;
			}
		}

		for (int t = 0; t < seqlen; ++t) {
			const float x_t = static_cast<float>(
					x_ptr[batch_id * x_batch_stride + channel_id * x_c_stride + t * x_l_stride]);

			if (!circular) {
				const int write_idx = state_len - advance_len + t;
				if (t < advance_len && write_idx >= 0) {
					conv_state_channel[write_idx * cs_l_stride] = static_cast<scalar_t>(x_t);
				}
			} else {
				conv_state_channel[update_idx * cs_l_stride] = static_cast<scalar_t>(x_t);
				++update_idx;
				update_idx = (update_idx >= state_len) ? (update_idx - state_len) : update_idx;
			}

			x_vals[kWidth - 1] = x_t;

			float out_val = bias_val;
#pragma unroll
			for (int w = 0; w < kWidth; ++w) {
				out_val += wt[w] * x_vals[w];
			}
			if (silu_activation) {
				out_val = out_val / (1.f + sycl::exp(-out_val));
			}

			out_ptr[batch_id * out_batch_stride + channel_id * out_c_stride + t * out_l_stride] =
					static_cast<scalar_t>(out_val);

#pragma unroll
			for (int w = 0; w < kWidth - 1; ++w) {
				x_vals[w] = x_vals[w + 1];
			}
		}
	}
};

template <typename scalar_t>
static void launch_causal_conv1d_fwd(
		scalar_t* x, scalar_t* w, scalar_t* bias, scalar_t* conv_states,
		const int* query_start_loc, const bool* has_initial_state, const int* cache_indices,
		scalar_t* out,
		int batch, int dim, int seqlen, int width, bool varlen, bool silu_activation, int64_t pad_slot_id,
		int x_batch_stride, int x_c_stride, int x_l_stride,
		int w_c_stride, int w_w_stride,
		int cs_batch_stride, int cs_c_stride, int cs_l_stride,
		int out_batch_stride, int out_c_stride, int out_l_stride,
		sycl::queue& q) {
	constexpr int kNThreads = 128;
	const int channel_groups = (dim + kNThreads - 1) / kNThreads;
	sycl::nd_range<2> range(
			sycl::range<2>(batch, channel_groups * kNThreads),
			sycl::range<2>(1, kNThreads));

#define LAUNCH_WIDTH(W)                                                                       \
	if (width == W) {                                                                           \
		CausalConv1dFwdKernel<scalar_t, W> kern{                                                  \
				x, w, bias, conv_states,                                                              \
				query_start_loc, has_initial_state, cache_indices, out,                               \
				batch, dim, seqlen, pad_slot_id, silu_activation, varlen,                             \
				x_batch_stride, x_c_stride, x_l_stride,                                               \
				w_c_stride, w_w_stride,                                                               \
				cs_batch_stride, cs_c_stride, cs_l_stride,                                            \
				out_batch_stride, out_c_stride, out_l_stride,                                         \
				width - 1};                                                                           \
		q.submit([&](sycl::handler& h) { h.parallel_for(range, kern); });                        \
		return;                                                                                   \
	}

	LAUNCH_WIDTH(2)
	LAUNCH_WIDTH(3)
	LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
	TORCH_CHECK(false, "causal_conv1d_fwd: unsupported width ", width);
}

template <typename scalar_t>
static void launch_causal_conv1d_update(
		scalar_t* x, scalar_t* w, scalar_t* bias, scalar_t* conv_state,
		const int* conv_state_indices, const int* cache_seqlens, scalar_t* out,
		int batch, int dim, int seqlen, int width, bool silu_activation, int64_t pad_slot_id,
		int x_batch_stride, int x_c_stride, int x_l_stride,
		int w_c_stride, int w_w_stride,
		int cs_batch_stride, int cs_c_stride, int cs_l_stride,
		int out_batch_stride, int out_c_stride, int out_l_stride,
		sycl::queue& q) {
	constexpr int kNThreads = 64;
	const int channel_groups = (dim + kNThreads - 1) / kNThreads;
	sycl::nd_range<2> range(
			sycl::range<2>(batch, channel_groups * kNThreads),
			sycl::range<2>(1, kNThreads));

#define LAUNCH_WIDTH(W)                                                                       \
	if (width == W) {                                                                           \
		CausalConv1dUpdateKernel<scalar_t, W> kern{                                               \
				x, w, bias, conv_state, conv_state_indices, cache_seqlens, out,                       \
				batch, dim, seqlen, width - 1, pad_slot_id, silu_activation,                          \
				x_batch_stride, x_c_stride, x_l_stride,                                               \
				w_c_stride, w_w_stride,                                                               \
				cs_batch_stride, cs_c_stride, cs_l_stride,                                            \
				out_batch_stride, out_c_stride, out_l_stride};                                        \
		q.submit([&](sycl::handler& h) { h.parallel_for(range, kern); });                        \
		return;                                                                                   \
	}

	LAUNCH_WIDTH(2)
	LAUNCH_WIDTH(3)
	LAUNCH_WIDTH(4)
#undef LAUNCH_WIDTH
	TORCH_CHECK(false, "causal_conv1d_update: unsupported width ", width);
}

void causal_conv1d_fwd(
		const at::Tensor& x,
		const at::Tensor& weight,
		const std::optional<at::Tensor>& bias_,
		const std::optional<at::Tensor>& conv_states,
		const std::optional<at::Tensor>& query_start_loc,
		const std::optional<at::Tensor>& cache_indices,
		const std::optional<at::Tensor>& has_initial_state,
		bool silu_activation,
		int64_t pad_slot_id) {
	auto dtype = x.scalar_type();
	TORCH_CHECK(
			dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Float,
			"causal_conv1d_fwd: unsupported dtype");
	TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
	TORCH_CHECK(weight.is_xpu(), "weight must be an XPU tensor");

	const bool varlen = query_start_loc.has_value();
	const auto sizes = x.sizes();
	const int batch_size = varlen ? (int)(query_start_loc.value().size(0) - 1) : (int)sizes[0];
	const int dim = varlen ? (int)sizes[0] : (int)sizes[1];
	const int seqlen = varlen ? (int)sizes[1] : (int)sizes[2];
	const int width = (int)weight.size(-1);

	if (varlen) {
		CHECK_SHAPE(x, dim, seqlen);
	} else {
		CHECK_SHAPE(x, batch_size, dim, seqlen);
	}
	CHECK_SHAPE(weight, dim, width);

	TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d_fwd: width must be 2, 3, or 4");

	if (bias_.has_value()) {
		const auto& bias = bias_.value();
		TORCH_CHECK(bias.scalar_type() == weight.scalar_type(), "bias dtype must match weight dtype");
		TORCH_CHECK(bias.is_xpu(), "bias must be an XPU tensor");
		TORCH_CHECK(bias.stride(-1) == 1, "bias must be contiguous on last dim");
		CHECK_SHAPE(bias, dim);
	}

	if (has_initial_state.has_value()) {
		const auto& his = has_initial_state.value();
		TORCH_CHECK(his.scalar_type() == at::ScalarType::Bool, "has_initial_state must be bool");
		TORCH_CHECK(his.is_xpu(), "has_initial_state must be an XPU tensor");
		CHECK_SHAPE(his, batch_size);
	}

	if (query_start_loc.has_value()) {
		const auto& qsl = query_start_loc.value();
		TORCH_CHECK(qsl.scalar_type() == at::ScalarType::Int, "query_start_loc must be int32");
		TORCH_CHECK(qsl.is_xpu(), "query_start_loc must be an XPU tensor");
	}

	if (cache_indices.has_value()) {
		const auto& ci = cache_indices.value();
		TORCH_CHECK(ci.scalar_type() == at::ScalarType::Int, "cache_indices must be int32");
		TORCH_CHECK(ci.is_xpu(), "cache_indices must be an XPU tensor");
		CHECK_SHAPE(ci, batch_size);
	}

	if (conv_states.has_value()) {
		const auto& cs = conv_states.value();
		TORCH_CHECK(cs.scalar_type() == dtype, "conv_states dtype must match x dtype");
		TORCH_CHECK(cs.is_xpu(), "conv_states must be an XPU tensor");
	}

	at::Tensor out = x;
	auto q = c10::xpu::getCurrentXPUStream().queue();

	const int* qsl_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr<int>() : nullptr;
	const bool* his_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr<bool>() : nullptr;
	const int* ci_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr<int>() : nullptr;

	int cs_bs = 0, cs_cs = 0, cs_ls = 0;
	void* cs_ptr_raw = nullptr;
	if (conv_states.has_value()) {
		const auto& cs = conv_states.value();
		cs_bs = (int)cs.stride(0);
		cs_cs = (int)cs.stride(-2);
		cs_ls = (int)cs.stride(-1);
		cs_ptr_raw = cs.data_ptr();
	}

	const int x_bs = varlen ? (int)x.stride(1) : (int)x.stride(0);
	const int x_cs = varlen ? (int)x.stride(0) : (int)x.stride(1);
	const int x_ls = varlen ? 0 : (int)x.stride(2);

	const int out_bs = x_bs;
	const int out_cs = x_cs;
	const int out_ls = x_ls;
	const int w_cs = (int)weight.stride(0);
	const int w_ws = (int)weight.stride(1);

#define DISPATCH_DTYPE(DTYPE, scalar_t)                                                          \
	if (dtype == DTYPE) {                                                                          \
		launch_causal_conv1d_fwd<scalar_t>(                                                          \
				reinterpret_cast<scalar_t*>(x.data_ptr()),                                               \
				reinterpret_cast<scalar_t*>(weight.data_ptr()),                                          \
				bias_.has_value() ? reinterpret_cast<scalar_t*>(bias_.value().data_ptr()) : nullptr,    \
				reinterpret_cast<scalar_t*>(cs_ptr_raw),                                                 \
				qsl_ptr, his_ptr, ci_ptr,                                                                \
				reinterpret_cast<scalar_t*>(out.data_ptr()),                                             \
				batch_size, dim, seqlen, width, varlen, silu_activation, pad_slot_id,                   \
				x_bs, x_cs, x_ls, w_cs, w_ws,                                                            \
				cs_bs, cs_cs, cs_ls,                                                                     \
				out_bs, out_cs, out_ls, q);                                                              \
	}

	DISPATCH_DTYPE(at::ScalarType::Half, at::Half)
	DISPATCH_DTYPE(at::ScalarType::BFloat16, at::BFloat16)
	DISPATCH_DTYPE(at::ScalarType::Float, float)
#undef DISPATCH_DTYPE
}

void causal_conv1d_update(
		const at::Tensor& x,
		const at::Tensor& conv_state,
		const at::Tensor& weight,
		const std::optional<at::Tensor>& bias_,
		bool silu_activation,
		const std::optional<at::Tensor>& cache_seqlens_,
		const std::optional<at::Tensor>& conv_state_indices_,
		int64_t pad_slot_id) {
	auto dtype = x.scalar_type();
	TORCH_CHECK(
			dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Float,
			"causal_conv1d_update: unsupported dtype");
	TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
	TORCH_CHECK(conv_state.is_xpu(), "conv_state must be an XPU tensor");
	TORCH_CHECK(weight.is_xpu(), "weight must be an XPU tensor");

	const int batch_size = (int)x.size(0);
	const int dim = (int)x.size(1);
	const int seqlen = (int)x.size(2);
	const int width = (int)weight.size(-1);
	const int conv_state_len = (int)conv_state.size(2);

	CHECK_SHAPE(x, batch_size, dim, seqlen);
	CHECK_SHAPE(weight, dim, width);
	TORCH_CHECK(conv_state_len >= width - 1, "conv_state_len must be >= width - 1");
	TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d_update: width must be 2, 3, or 4");

	if (bias_.has_value()) {
		const auto& bias = bias_.value();
		TORCH_CHECK(bias.scalar_type() == weight.scalar_type(), "bias dtype must match weight dtype");
		TORCH_CHECK(bias.is_xpu(), "bias must be an XPU tensor");
		TORCH_CHECK(bias.stride(-1) == 1, "bias must be contiguous on last dim");
		CHECK_SHAPE(bias, dim);
	}

	if (cache_seqlens_.has_value()) {
		const auto& cache_seqlens = cache_seqlens_.value();
		TORCH_CHECK(cache_seqlens.scalar_type() == at::ScalarType::Int, "cache_seqlens must be int32");
		TORCH_CHECK(cache_seqlens.is_xpu(), "cache_seqlens must be an XPU tensor");
		TORCH_CHECK(cache_seqlens.stride(-1) == 1, "cache_seqlens must be contiguous");
		CHECK_SHAPE(cache_seqlens, batch_size);
	}

	if (conv_state_indices_.has_value()) {
		const auto& conv_state_indices = conv_state_indices_.value();
		TORCH_CHECK(conv_state_indices.scalar_type() == at::ScalarType::Int, "conv_state_indices must be int32");
		TORCH_CHECK(conv_state_indices.is_xpu(), "conv_state_indices must be an XPU tensor");
		TORCH_CHECK(conv_state_indices.stride(0) == 1, "conv_state_indices must be contiguous");
		CHECK_SHAPE(conv_state_indices, batch_size);
	}

	at::Tensor out = x;
	auto q = c10::xpu::getCurrentXPUStream().queue();

	const int* csi_ptr = conv_state_indices_.has_value() ? conv_state_indices_.value().data_ptr<int>() : nullptr;
	const int* cache_seqlens_ptr = cache_seqlens_.has_value() ? cache_seqlens_.value().data_ptr<int>() : nullptr;

#define DISPATCH_DTYPE(DTYPE, scalar_t)                                                          \
	if (dtype == DTYPE) {                                                                          \
		launch_causal_conv1d_update<scalar_t>(                                                       \
				reinterpret_cast<scalar_t*>(x.data_ptr()),                                               \
				reinterpret_cast<scalar_t*>(weight.data_ptr()),                                          \
				bias_.has_value() ? reinterpret_cast<scalar_t*>(bias_.value().data_ptr()) : nullptr,    \
				reinterpret_cast<scalar_t*>(conv_state.data_ptr()),                                      \
				csi_ptr, cache_seqlens_ptr,                                                              \
				reinterpret_cast<scalar_t*>(out.data_ptr()),                                             \
				batch_size, dim, seqlen, width, silu_activation, pad_slot_id,                           \
				(int)x.stride(0), (int)x.stride(1), (int)x.stride(2),                                   \
				(int)weight.stride(0), (int)weight.stride(1),                                            \
				(int)conv_state.stride(0), (int)conv_state.stride(1), (int)conv_state.stride(2),        \
				(int)out.stride(0), (int)out.stride(1), (int)out.stride(2), q);                         \
	}

	DISPATCH_DTYPE(at::ScalarType::Half, at::Half)
	DISPATCH_DTYPE(at::ScalarType::BFloat16, at::BFloat16)
	DISPATCH_DTYPE(at::ScalarType::Float, float)
#undef DISPATCH_DTYPE
}

