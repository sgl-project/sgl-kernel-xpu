#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/core/Array.h>

#include "Utils.h"
#include "comm/General.h"

namespace at::native::xpu {

enum class EmbeddingAlgorithm { RotateHalf = 0, RotateInterleave = 1 };

template <typename scalar_t, EmbeddingAlgorithm algo = EmbeddingAlgorithm::RotateHalf>
inline void apply_token_rotary_embedding(
    scalar_t* data,
    const scalar_t* cos_cache,
    const scalar_t* sin_cache,
    int rot_offset,
    [[maybe_unused]] int embed_dim) {
  using accscalar = at::opmath_type<scalar_t>;
  int x_idx, y_idx;
  accscalar cos_val, sin_val;
  if constexpr (algo == EmbeddingAlgorithm::RotateHalf) {
    x_idx = rot_offset;
    y_idx = rot_offset + embed_dim;
    cos_val = cos_cache[x_idx];
    sin_val = sin_cache[x_idx];
  } else {
    x_idx = 2 * rot_offset;
    y_idx = 2 * rot_offset + 1;
    cos_val = cos_cache[x_idx / 2];
    sin_val = sin_cache[x_idx / 2];
  }
  accscalar x = data[x_idx];
  accscalar y = data[y_idx];
  data[x_idx] = x * cos_val - y * sin_val;
  data[y_idx] = x * sin_val + y * cos_val;
}

template <typename scalar_t, EmbeddingAlgorithm algo = EmbeddingAlgorithm::RotateHalf, bool has_cache_offset = true>
struct RotaryEmbeddingBatched {
  void operator()(sycl::nd_item<1> item) const {
    using accscalar = at::opmath_type<scalar_t>;
    int32_t token_id = item.get_group(0);
    int32_t start_idx = item.get_local_id(0);
    int64_t pos = positions_[token_id];
    int64_t cos_sin_offset = 0;
    if constexpr (has_cache_offset) cos_sin_offset = cos_sin_cache_offsets_[token_id];
    scalar_t* cos_cache = cos_sin_cache_ + (pos + cos_sin_offset) * rot_dim_;
    int32_t embed_dim = rot_dim_ / 2;
    scalar_t* sin_cache = cos_cache + embed_dim;
    scalar_t* query_base = query_ + token_id * query_stride_;
    scalar_t* key_base = key_ + token_id * key_stride_;
    int32_t q_num = num_heads_ * embed_dim;
    int32_t k_num = num_kv_heads_ * embed_dim;
    int32_t local_range = item.get_local_range(0);
    for (int i = start_idx; i < q_num; i += local_range) {
      int32_t head_id = i / embed_dim;
      int32_t rot_offset = i % embed_dim;
      scalar_t* query_st = query_base + head_id * head_size_;

      apply_token_rotary_embedding<scalar_t, algo>(query_st, cos_cache, sin_cache, rot_offset, embed_dim);
    }

    for (int i = start_idx; i < k_num; i += local_range) {
      int32_t head_id = i / embed_dim;
      int32_t rot_offset = i % embed_dim;
      scalar_t* key_st = key_base + head_id * head_size_;
      apply_token_rotary_embedding<scalar_t, algo>(key_st, cos_cache, sin_cache, rot_offset, embed_dim);
    }
  }

  int64_t* positions_;
  scalar_t* cos_sin_cache_;
  int64_t* cos_sin_cache_offsets_;
  scalar_t* query_;
  scalar_t* key_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t query_stride_;
  int64_t key_stride_;
  int64_t head_size_;
  int64_t rot_dim_;
};

namespace DSRotaryEmbedding {
template <typename T, int64_t rotary_dim, bool is_neox>
struct FusedDSRotaryEmbeddingQK {
  static constexpr int sg_size = 16;
  static constexpr int64_t sg_no = 1;
  FusedDSRotaryEmbeddingQK(
      const int64_t* positions,
      const T* query,
      const T* key,
      const int64_t* offsets,
      const T* cos_sin_cache,
      T* query_out,
      T* key_out,
      const int64_t batch,
      const int64_t q_num_head,
      const int64_t k_num_head,
      const int64_t head_size,
      const int64_t q_num_head_d,
      const int64_t q_batch_d,
      const int64_t k_num_head_d,
      const int64_t k_batch_d)
      : positions(positions),
        query(query),
        key(key),
        offsets(offsets),
        cos_sin_cache(cos_sin_cache),
        query_out(query_out),
        key_out(key_out),
        batch(batch),
        q_num_head(q_num_head),
        k_num_head(k_num_head),
        head_size(head_size),
        q_num_head_d(q_num_head_d),
        q_batch_d(q_batch_d),
        k_num_head_d(k_num_head_d),
        k_batch_d(k_batch_d) {}

  static inline sycl::nd_range<3>
  get_nd_range(const int64_t batch, const int64_t q_num_head, const int64_t k_num_head) {
    const int64_t sg_per_heads = divup(q_num_head + k_num_head, sg_size);
    // const int64_t thd_per_heads = sg_per_heads * sg_size;
    sycl::range<3> local(1, sg_per_heads, sg_size);
    sycl::range<3> global(batch, sg_per_heads, sg_size);
    return sycl::nd_range<3>(global, local);
  }

  void rotary_emb_kern(const int64_t position, const T* pe, const T* cos_sin_cache, T* res) const {
    constexpr int64_t half_rotary_dim = rotary_dim / 2;
    constexpr int64_t vec_2_len = 2;
    using v2_type = sycl::vec<T, vec_2_len>;
    const int64_t cache_idx = position * rotary_dim;
    const T* cos_cache_offset = &cos_sin_cache[cache_idx];
    const T* sin_cache_offset = cos_cache_offset + half_rotary_dim;
    if constexpr (is_neox) {
      // repeat & rotate mul add
      for (int64_t i = 0; i < half_rotary_dim; ++i) {
        int64_t j = i + half_rotary_dim;
        T cv = cos_cache_offset[i];
        T sv = sin_cache_offset[i];
        res[i] = pe[i] * cv - pe[j] * sv;
        res[j] = pe[j] * cv + pe[i] * sv;
      }
    } else {
      // interleave & rotate mul add, unfortunately no prefetch in sycl
      const v2_type* pe_2 = reinterpret_cast<const v2_type*>(pe);
      v2_type* res_2 = reinterpret_cast<v2_type*>(res);
      for (int64_t h = 0; h < half_rotary_dim; ++h) {
        T c = cos_cache_offset[h];
        T s = sin_cache_offset[h];
        v2_type c2 = {c, c};
        v2_type s2 = {s, s};
        v2_type t = pe_2[h];
        v2_type* dst = &res_2[h];
        v2_type tr = {-t[1], t[0]};
        *dst = t * c2 + tr * s2;
      }
    }
  }

  [[sycl::reqd_sub_group_size(sg_size)]] void operator()(sycl::nd_item<3> idx) const {
    int64_t batch_idx = idx.get_global_id(0);
    int64_t sg_idx = idx.get_local_id(1);
    int64_t local_id = idx.get_global_id(2);
    int64_t head_idx = sg_idx * sg_size + local_id;
    int64_t qo_idx = batch_idx * q_num_head * head_size + head_idx * head_size;
    int64_t ko_idx = batch_idx * k_num_head * head_size + (head_idx - q_num_head) * head_size;
    int64_t qi_idx = batch_idx * q_batch_d + head_idx * q_num_head_d;
    int64_t ki_idx = batch_idx * k_batch_d + (head_idx - q_num_head) * k_num_head_d;
    if (head_idx < q_num_head) {
      rotary_emb_kern(positions[batch_idx], &query[qi_idx], cos_sin_cache, &query_out[qo_idx]);
    } else if (head_idx < q_num_head + k_num_head) {
      rotary_emb_kern(positions[batch_idx], &key[ki_idx], cos_sin_cache, &key_out[ko_idx]);
    }
  }

  const int64_t* positions;
  const T* query;
  const T* key;
  const int64_t* offsets;
  const T* cos_sin_cache;
  T* query_out;
  T* key_out;
  const int64_t batch;
  const int64_t q_num_head;
  const int64_t k_num_head;
  const int64_t head_size;
  const int64_t q_num_head_d;
  const int64_t q_batch_d;
  const int64_t k_num_head_d;
  const int64_t k_batch_d;
};

template <typename T, int64_t rotary_dim, bool is_neox>
void launch_rotary_embedding(
    sycl::queue& Q,
    const int64_t* positions,
    const T* query,
    const T* key,
    const int64_t* offsets,
    const T* cos_sin_cache,
    T* query_out,
    T* key_out,
    const int64_t batch,
    const int64_t q_num_head,
    const int64_t k_num_head,
    const int64_t head_size,
    const int64_t q_num_head_d,
    const int64_t q_batch_d,
    const int64_t k_num_head_d,
    const int64_t k_batch_d) {
  using Kernel = FusedDSRotaryEmbeddingQK<T, rotary_dim, is_neox>;
  auto range = Kernel::get_nd_range(batch, q_num_head, k_num_head);
  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        positions,
        query,
        key,
        offsets,
        cos_sin_cache,
        query_out,
        key_out,
        batch,
        q_num_head,
        k_num_head,
        head_size,
        q_num_head_d,
        q_batch_d,
        k_num_head_d,
        k_batch_d);
    cgh.parallel_for(range, task);
  };
  Q.submit(cgf);
}

template <typename T>
using LAUNCH_FUNC = void (*)(
    sycl::queue&,
    const int64_t*,
    const T*,
    const T*,
    const int64_t*,
    const T*,
    T*,
    T*,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t);

#define DEFINE_DS_ROTEMB_FUNC(T, n, b) &launch_rotary_embedding<T, n, b>

template <typename T>
void launch_rotary_embedding(
    const int64_t* positions,
    const T* query,
    const T* key,
    const int64_t* offsets,
    const T* cos_sin_cache,
    T* query_out,
    T* key_out,
    const int64_t batch,
    const int64_t q_num_head,
    const int64_t k_num_head,
    const int64_t head_size,
    const int64_t rotary_dim,
    bool is_neox_style,
    const int64_t q_num_head_d,
    const int64_t q_batch_d,
    const int64_t k_num_head_d,
    const int64_t k_batch_d) {
  auto& queue = dpcppGetCurrentQueue();

  constexpr int dim_size = 5;
  constexpr std::array<int, dim_size> allowed_dim = {32, 64, 96, 128, 256};
  int rot_idx = -1;
  int neox_idx = is_neox_style ? 1 : 0;
  for (int i = 0; i < allowed_dim.size(); ++i) {
    if (allowed_dim[i] == rotary_dim) {
      rot_idx = i;
    }
  }
  TORCH_CHECK(rot_idx >= 0, "wrong values for rotary_dim (%ld) only support 32,64,96,128,256\n", rotary_dim);
  TORCH_CHECK(rotary_dim == head_size, "rotary_dim (%ld)should be equal to head_size (%ld)", rotary_dim, head_size);
  int funcIndex = neox_idx * allowed_dim.size() + rot_idx;
  constexpr int func_size = dim_size * 2;
  static constexpr std::array<LAUNCH_FUNC<T>, func_size> launch_funcs = {
      DEFINE_DS_ROTEMB_FUNC(T, 32, false),
      DEFINE_DS_ROTEMB_FUNC(T, 64, false),
      DEFINE_DS_ROTEMB_FUNC(T, 96, false),
      DEFINE_DS_ROTEMB_FUNC(T, 128, false),
      DEFINE_DS_ROTEMB_FUNC(T, 256, false),
      DEFINE_DS_ROTEMB_FUNC(T, 32, true),
      DEFINE_DS_ROTEMB_FUNC(T, 64, true),
      DEFINE_DS_ROTEMB_FUNC(T, 96, true),
      DEFINE_DS_ROTEMB_FUNC(T, 128, true),
      DEFINE_DS_ROTEMB_FUNC(T, 256, true),
  };
  launch_funcs[funcIndex](
      queue,
      positions,
      query,
      key,
      offsets,
      cos_sin_cache,
      query_out,
      key_out,
      batch,
      q_num_head,
      k_num_head,
      head_size,
      q_num_head_d,
      q_batch_d,
      k_num_head_d,
      k_batch_d);
}
}  // namespace DSRotaryEmbedding

void rotary_embedding_2D_kernel_impl(
    const at::Tensor& positions,  //[batch_size, seqlen] or [num_tokens]
    const at::Tensor& query,      // [(bs, seq)/num_tokens, num_head * head_dim]
    const at::Tensor& key,        // [(bs, seq)/num_tokens, num_kv_head * head_dim]
    int64_t head_size,
    const at::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox,
    int64_t rot_dim) {
  int64_t num_tokens = positions.view(-1).size(0);
  int64_t num_heads = query.size(-1) / head_size;
  int64_t num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  auto queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t max_group_num = dpcppMaxWorkItemsPerTile(dev_id) / max_wg_size;
  int64_t num_groups = num_tokens;
  int64_t num_eus = dpcppGpuEuCount(dev_id);
  int64_t group_size = std::min(max_wg_size, query.size(-1));

  if (num_tokens >= num_eus) {
    group_size = std::min<int64_t>(std::min<int64_t>(num_heads * rot_dim / 2, 512), max_wg_size);
  }

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "rotary_embedding_2D_kernel_impl", [=]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (is_neox) {
            RotaryEmbeddingBatched<scalar_t, EmbeddingAlgorithm::RotateHalf, false> kernel = {
                .positions_ = positions.data_ptr<int64_t>(),
                .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                .cos_sin_cache_offsets_ = nullptr,
                .query_ = query.data_ptr<scalar_t>(),
                .key_ = key.data_ptr<scalar_t>(),
                .num_heads_ = num_heads,
                .num_kv_heads_ = num_kv_heads,
                .query_stride_ = query_stride,
                .key_stride_ = key_stride,
                .head_size_ = head_size,
                .rot_dim_ = rot_dim,
            };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)), kernel);
          } else {
            RotaryEmbeddingBatched<scalar_t, EmbeddingAlgorithm::RotateInterleave, false> kernel = {
                .positions_ = positions.data_ptr<int64_t>(),
                .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                .cos_sin_cache_offsets_ = nullptr,
                .query_ = query.data_ptr<scalar_t>(),
                .key_ = key.data_ptr<scalar_t>(),
                .num_heads_ = num_heads,
                .num_kv_heads_ = num_kv_heads,
                .query_stride_ = query_stride,
                .key_stride_ = key_stride,
                .head_size_ = head_size,
                .rot_dim_ = rot_dim,
            };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)), kernel);
          }
        };
        dpcppGetCurrentQueue().submit(cgf);
      });
}

/**
 * @brief Perform deepseek rotary embedding with q&k.
 * @param positions index of embedding [batch]
 * @param query query to be processed [batch, num_head, head_dim]
 * @param key key to be processed [batch, num_head, head_dim]
 * @param cos_sin_cache shared cache with cos/sin
 * @param is_neox_style choose interleave or half.
 * @return A tuple of tensors (query_out, key_out).
 */
std::tuple<at::Tensor, at::Tensor> rotary_embedding_3D_kernel_impl(
    const at::Tensor& positions,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox_style) {
  auto query_out = at::empty_like(query);
  auto key_out = at::empty_like(key);

  auto q_shape = query.sizes();
  auto q_stride = query.strides();
  int64_t head_size = q_shape[2];
  int64_t q_num_head = q_shape[1];
  int64_t batch = q_shape[0];
  int64_t q_num_head_d = q_stride[1];
  int64_t q_batch_d = q_stride[0];
  auto k_shape = key.sizes();
  auto k_stride = key.strides();
  int64_t k_num_head = k_shape[1];
  int64_t k_num_head_d = k_stride[1];
  int64_t k_batch_d = k_stride[0];
  if (is_neox_style) {
    query_out = query_out.reshape({1, batch, q_num_head, head_size});
    key_out = key_out.reshape({1, batch, k_num_head, head_size});
  }
  TORCH_CHECK(cos_sin_cache.sizes()[1] == head_size, "Rotary dim doesn't match query head_size");
  TORCH_CHECK(cos_sin_cache.sizes()[1] == k_shape[2], "Rotary dim doesn't match key head_size");
  int64_t* offsets_ptr = nullptr;
  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, query.scalar_type(), "rotary_embedding_3D_kernel_impl", [&]() {
        DSRotaryEmbedding::launch_rotary_embedding<scalar_t>(
            reinterpret_cast<int64_t*>(positions.data_ptr()),
            reinterpret_cast<scalar_t*>(query.data_ptr()),
            reinterpret_cast<scalar_t*>(key.data_ptr()),
            reinterpret_cast<int64_t*>(offsets_ptr),
            reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<scalar_t*>(query_out.data_ptr()),
            reinterpret_cast<scalar_t*>(key_out.data_ptr()),
            batch,
            q_num_head,
            k_num_head,
            head_size,
            rotary_dim,
            is_neox_style,
            q_num_head_d,
            q_batch_d,
            k_num_head_d,
            k_batch_d);
      });

  return {query_out, key_out};
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
  const auto input_dim = query.dim();
  int64_t rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(
      input_dim == 2 || input_dim == 3,
      " Query/Key must be 2D [num_tokens, num_heads*head_size] or 3D [num_tokens, num_heads, head_size] tensor");
  if (input_dim == 2) {
    rotary_embedding_2D_kernel_impl(positions, query, key, head_size, cos_sin_cache, is_neox, rotary_dim);
    return {query, key};
  } else {
    return rotary_embedding_3D_kernel_impl(positions, query, key, cos_sin_cache, rotary_dim, is_neox);
  }
}

// ============================================================================
// Fused RoPE + KV-cache write kernel
// Mirrors CUDA `apply_rope_inplace_with_kvcache` from jit_kernel/rope.py:140-175.
// Single kernel: rotates Q/K in-place (fp32 arithmetic) and writes rotated K + V
// into the flat KV-cache at out_loc.
// ============================================================================

template <typename scalar_t, EmbeddingAlgorithm algo>
struct FusedRopeKVCacheKernel {
  using accscalar = at::opmath_type<scalar_t>;

  void operator()(sycl::nd_item<1> item) const {
    int32_t token_id = item.get_group(0);
    int32_t local_id = item.get_local_id(0);
    int32_t local_range = item.get_local_range(0);

    int64_t out_idx = out_loc_[token_id];
    if (out_idx < 0) return;  // speculative decoding skip

    int64_t pos = positions_[token_id];
    int32_t embed_dim = rot_dim_ / 2;

    // cos_sin_cache is float32; index by position
    const float* cos_cache = cos_sin_cache_ + pos * rot_dim_;
    const float* sin_cache = cos_cache + embed_dim;

    // RoPE Q in-place
    scalar_t* q_base = query_ + token_id * num_q_heads_ * head_dim_;
    int32_t q_work = num_q_heads_ * embed_dim;
    for (int i = local_id; i < q_work; i += local_range) {
      int32_t head_id = i / embed_dim;
      int32_t rot_offset = i % embed_dim;
      scalar_t* q_head = q_base + head_id * head_dim_;

      int x_idx, y_idx;
      float cos_val, sin_val;
      if constexpr (algo == EmbeddingAlgorithm::RotateHalf) {
        x_idx = rot_offset;
        y_idx = rot_offset + embed_dim;
        cos_val = cos_cache[rot_offset];
        sin_val = sin_cache[rot_offset];
      } else {
        x_idx = 2 * rot_offset;
        y_idx = 2 * rot_offset + 1;
        cos_val = cos_cache[rot_offset];
        sin_val = sin_cache[rot_offset];
      }
      accscalar x = static_cast<accscalar>(q_head[x_idx]);
      accscalar y = static_cast<accscalar>(q_head[y_idx]);
      q_head[x_idx] = static_cast<scalar_t>(x * cos_val - y * sin_val);
      q_head[y_idx] = static_cast<scalar_t>(x * sin_val + y * cos_val);
    }

    // RoPE K in-place + write to k_cache
    scalar_t* k_base = key_ + token_id * num_kv_heads_ * head_dim_;
    scalar_t* k_cache_row = k_cache_ + out_idx * num_kv_heads_ * head_dim_;
    int32_t k_work = num_kv_heads_ * embed_dim;
    for (int i = local_id; i < k_work; i += local_range) {
      int32_t head_id = i / embed_dim;
      int32_t rot_offset = i % embed_dim;
      scalar_t* k_head = k_base + head_id * head_dim_;

      int x_idx, y_idx;
      float cos_val, sin_val;
      if constexpr (algo == EmbeddingAlgorithm::RotateHalf) {
        x_idx = rot_offset;
        y_idx = rot_offset + embed_dim;
        cos_val = cos_cache[rot_offset];
        sin_val = sin_cache[rot_offset];
      } else {
        x_idx = 2 * rot_offset;
        y_idx = 2 * rot_offset + 1;
        cos_val = cos_cache[rot_offset];
        sin_val = sin_cache[rot_offset];
      }
      accscalar x = static_cast<accscalar>(k_head[x_idx]);
      accscalar y = static_cast<accscalar>(k_head[y_idx]);
      scalar_t x_rot = static_cast<scalar_t>(x * cos_val - y * sin_val);
      scalar_t y_rot = static_cast<scalar_t>(x * sin_val + y * cos_val);
      k_head[x_idx] = x_rot;
      k_head[y_idx] = y_rot;
      // Write rotated K to cache
      k_cache_row[head_id * head_dim_ + x_idx] = x_rot;
      k_cache_row[head_id * head_dim_ + y_idx] = y_rot;
    }

    // Copy non-rotated dims of K to cache (if head_dim > rot_dim)
    // and write full V to v_cache
    scalar_t* v_base = value_ + token_id * num_kv_heads_ * head_dim_;
    scalar_t* v_cache_row = v_cache_ + out_idx * num_kv_heads_ * head_dim_;
    int32_t full_work = num_kv_heads_ * head_dim_;
    for (int i = local_id; i < full_work; i += local_range) {
      v_cache_row[i] = v_base[i];
    }

    // For head_dim > rot_dim: copy the non-rotated tail of K to cache
    if (head_dim_ > rot_dim_) {
      for (int i = local_id; i < num_kv_heads_ * (head_dim_ - rot_dim_); i += local_range) {
        int32_t head_id = i / (head_dim_ - rot_dim_);
        int32_t dim_offset = i % (head_dim_ - rot_dim_) + rot_dim_;
        k_cache_row[head_id * head_dim_ + dim_offset] = k_base[head_id * head_dim_ + dim_offset];
      }
    }
  }

  int64_t* positions_;
  float* cos_sin_cache_;  // always fp32
  scalar_t* query_;
  scalar_t* key_;
  scalar_t* value_;
  scalar_t* k_cache_;
  scalar_t* v_cache_;
  int64_t* out_loc_;
  int64_t num_q_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  int64_t rot_dim_;
};

void apply_rope_inplace_with_kvcache(
    at::Tensor& query,          // [num_tokens, n_q_heads, head_dim]
    at::Tensor& key,            // [num_tokens, n_kv_heads, head_dim]
    at::Tensor& value,          // [num_tokens, n_kv_heads, head_dim]
    at::Tensor& k_cache,        // [cache_size, n_kv_heads * head_dim]
    at::Tensor& v_cache,        // [cache_size, n_kv_heads * head_dim]
    at::Tensor& cos_sin_cache,  // [max_pos, rot_dim] — MUST be float32
    at::Tensor& positions,      // [num_tokens] int64
    at::Tensor& out_loc,        // [num_tokens] int64
    bool is_neox) {
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == at::ScalarType::Float,
      "cos_sin_cache must be float32, got ",
      cos_sin_cache.scalar_type());
  TORCH_CHECK(query.dim() == 3, "query must be 3D [num_tokens, n_heads, head_dim]");
  TORCH_CHECK(key.dim() == 3, "key must be 3D [num_tokens, n_kv_heads, head_dim]");
  TORCH_CHECK(value.dim() == 3, "value must be 3D [num_tokens, n_kv_heads, head_dim]");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous because the XPU RoPE kernel uses packed indexing");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous because the XPU RoPE kernel uses packed indexing");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous because the XPU RoPE kernel uses packed indexing");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous because the XPU RoPE kernel uses packed indexing");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous because the XPU RoPE kernel uses packed indexing");

  int64_t num_tokens = query.size(0);
  int64_t num_q_heads = query.size(1);
  int64_t num_kv_heads = key.size(1);
  int64_t head_dim = query.size(2);
  int64_t rot_dim = cos_sin_cache.size(1);

  TORCH_CHECK(key.size(2) == head_dim, "key head_dim must match query head_dim");
  TORCH_CHECK(value.size(1) == num_kv_heads, "value num_heads must match key");
  TORCH_CHECK(value.size(2) == head_dim, "value head_dim must match key head_dim");
  TORCH_CHECK(
      rot_dim <= head_dim,
      "rot_dim (",
      rot_dim,
      ") must be <= head_dim (",
      head_dim,
      "); cos_sin_cache.size(1) cannot exceed query head_dim");
  TORCH_CHECK(
      rot_dim % 2 == 0,
      "rot_dim (",
      rot_dim,
      ") must be even; the kernel pairs rotary components as (i, i+rot_dim/2) or (2i, 2i+1)");

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t max_work = std::max(num_q_heads * rot_dim / 2, num_kv_heads * head_dim);
  int64_t group_size = std::min<int64_t>(std::min<int64_t>(max_work, 512), max_wg_size);

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "apply_rope_inplace_with_kvcache", [&]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (is_neox) {
            FusedRopeKVCacheKernel<scalar_t, EmbeddingAlgorithm::RotateHalf> kernel = {
                .positions_ = positions.data_ptr<int64_t>(),
                .cos_sin_cache_ = cos_sin_cache.data_ptr<float>(),
                .query_ = query.data_ptr<scalar_t>(),
                .key_ = key.data_ptr<scalar_t>(),
                .value_ = value.data_ptr<scalar_t>(),
                .k_cache_ = k_cache.data_ptr<scalar_t>(),
                .v_cache_ = v_cache.data_ptr<scalar_t>(),
                .out_loc_ = out_loc.data_ptr<int64_t>(),
                .num_q_heads_ = num_q_heads,
                .num_kv_heads_ = num_kv_heads,
                .head_dim_ = head_dim,
                .rot_dim_ = rot_dim,
            };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(sycl::range<1>(num_tokens * group_size), sycl::range<1>(group_size)), kernel);
          } else {
            FusedRopeKVCacheKernel<scalar_t, EmbeddingAlgorithm::RotateInterleave> kernel = {
                .positions_ = positions.data_ptr<int64_t>(),
                .cos_sin_cache_ = cos_sin_cache.data_ptr<float>(),
                .query_ = query.data_ptr<scalar_t>(),
                .key_ = key.data_ptr<scalar_t>(),
                .value_ = value.data_ptr<scalar_t>(),
                .k_cache_ = k_cache.data_ptr<scalar_t>(),
                .v_cache_ = v_cache.data_ptr<scalar_t>(),
                .out_loc_ = out_loc.data_ptr<int64_t>(),
                .num_q_heads_ = num_q_heads,
                .num_kv_heads_ = num_kv_heads,
                .head_dim_ = head_dim,
                .rot_dim_ = rot_dim,
            };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(sycl::range<1>(num_tokens * group_size), sycl::range<1>(group_size)), kernel);
          }
        };
        dpcppGetCurrentQueue().submit(cgf);
      });
}

}  // namespace at::native::xpu
