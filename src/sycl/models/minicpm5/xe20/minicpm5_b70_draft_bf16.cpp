// SPDX-License-Identifier: Apache-2.0
#define MINICPM5_MONO_NAMESPACE b70_draft_bf16
#define MINICPM5_MONO_BF16_FP8_HALF_ENABLED
#define MINICPM5_MONO_DPAS_ENABLED
#define MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
#define MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
#include "launch.hpp"

namespace sgl_kernel::minicpm5::xe20::b70_draft_bf16 {
template <int TP, bool FP8>
class DraftLayer;

namespace {

constexpr int Rows = Shape::draft_rows;

inline UInt8 weight_tile(
    [[maybe_unused]] const uint8_t* weight,
    [[maybe_unused]] int pitch,
    [[maybe_unused]] int n,
    [[maybe_unused]] int col,
    [[maybe_unused]] int chunk) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
      intptr_t(weight), pitch - 1, n - 1, pitch - 1, Int2{chunk * 8, col});
#else
  return UInt8{};
#endif
}

template <bool FP8>
inline void accumulate_projection(Float8& sum, UInt8 weight, const uint32_t* packed, int chunk, int lane) {
  if constexpr (FP8) {
    // The same 32-byte block holds two K=16 tiles with FP8 weights.
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      UInt8 converted;
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        const uint32_t bytes = weight[tile * 4 + word];
        converted[2 * word] = fp8_pair_rows<Rows>(bytes);
        converted[2 * word + 1] = fp8_pair_rows<Rows>(bytes >> 16);
      }
      dpas16(sum, projection_a<Rows>(packed, 2 * chunk + tile, lane), converted);
    }
  } else {
    dpas16(sum, projection_a<Rows>(packed, chunk, lane), weight);
  }
}

enum class ProjectionOutput { Global, Local, SwiGlu };

template <bool FP8, int K, int N, int Slices, ProjectionOutput Output = ProjectionOutput::Global>
inline void
project(sycl::nd_item<1> item, DraftProjection weight, const uint32_t* packed, float* partial, scalar_t* output) {
  constexpr int GroupTiles = (N / 16 + Groups - 1) / Groups;
  constexpr int Pitch = K * (FP8 ? 1 : sizeof(scalar_t));
  constexpr int ChunkCount = Pitch / 32;
  constexpr int Plane = Subgroups * Rows * 16;
  constexpr bool SwiGlu = Output == ProjectionOutput::SwiGlu;
  static_assert(GroupTiles * Slices <= Subgroups && ChunkCount >= Slices);
  const auto sg = item.get_sub_group();
  const int subgroup = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int tile_base = item.get_group_linear_id() * GroupTiles;
  const int tile = tile_base + subgroup / Slices;
  const int slice = subgroup % Slices;
  if (subgroup < GroupTiles * Slices && tile < N / 16) {
    const int first = slice * ChunkCount / Slices;
    const int last = (slice + 1) * ChunkCount / Slices;
    Float8 sum{}, up_sum{};
    auto gate = weight_tile(weight.data, Pitch, N, tile * 16, first);
    UInt8 up{};
    if constexpr (SwiGlu) up = weight_tile(weight.data + size_t(N) * Pitch, Pitch, N, tile * 16, first);
    for (int chunk = first; chunk < last; ++chunk) {
      UInt8 next_gate{}, next_up{};
      if (chunk + 1 < last) {
        next_gate = weight_tile(weight.data, Pitch, N, tile * 16, chunk + 1);
        if constexpr (SwiGlu) next_up = weight_tile(weight.data + size_t(N) * Pitch, Pitch, N, tile * 16, chunk + 1);
      }
      accumulate_projection<FP8>(sum, gate, packed, chunk, lane);
      if constexpr (SwiGlu) accumulate_projection<FP8>(up_sum, up, packed, chunk, lane);
      gate = next_gate;
      up = next_up;
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
      const int index = (subgroup * Rows + row) * 16 + lane;
      partial[index] = sum[row];
      if constexpr (SwiGlu) partial[Plane + index] = up_sum[row];
    }
  }
  wg_bar(item.get_group());

  constexpr int Pair = SwiGlu ? 2 : 1;
  constexpr int ValuesPerTile = 16 / Pair;
  const int lid = item.get_local_linear_id();
  for (int job = lid; job < GroupTiles * Rows * ValuesPerTile; job += Local) {
    const int local_tile = job / (Rows * ValuesPerTile);
    const int out_tile = tile_base + local_tile;
    if (out_tile >= N / 16) continue;
    const int row = job / ValuesPerTile % Rows;
    const int col = (job % ValuesPerTile) * Pair;
    if constexpr (SwiGlu) {
      scalar_t values[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float g = 0, u = 0;
#pragma unroll
        for (int k = 0; k < Slices; ++k) {
          const int index = ((local_tile * Slices + k) * Rows + row) * 16 + col + i;
          g += partial[index];
          u += partial[Plane + index];
        }
        if constexpr (FP8) {
          const int channel = out_tile * 16 + col + i;
          g *= weight.scale[channel];
          u *= weight.scale[N + channel];
        }
        g = float(scalar_t(g));
        values[i] = scalar_t(g / (1.f + sycl::exp(-g)) * float(scalar_t(u)));
      }
      auto* packed_output = reinterpret_cast<uint32_t*>(output);
      packed_output[(out_tile * Rows + row) * 8 + col / 2] = pack16(values[0], values[1]);
    } else {
      float value = 0;
#pragma unroll
      for (int k = 0; k < Slices; ++k)
        value += partial[((local_tile * Slices + k) * Rows + row) * 16 + col];
      if constexpr (FP8) value *= weight.scale[out_tile * 16 + col];
      if constexpr (Output == ProjectionOutput::Local)
        output[row * (H / Groups) + local_tile * 16 + col] = scalar_t(value);
      else
        output[row * N + out_tile * 16 + col] = scalar_t(value);
    }
  }
}

inline void check_metadata(sycl::nd_item<1> item, Step step, uint32_t call) {
  const int id = item.get_global_linear_id();
  const int length = step.lengths[0];
  const int pages = length > 0 && length <= MaxContext ? (length + step.block_size - 1) / step.block_size : 0;
  if (id < Rows && step.slots[id] != -1) {
    const int64_t position = step.positions[id], slot = step.slots[id];
    bool valid = pages > 0 && pages <= step.block_table_cols && slot >= 0 &&
                 slot / step.block_size < step.cache_blocks && position >= 0 && position < length &&
                 position < step.rope_cache_rows;
    if (valid) {
      const auto page = step.blocks[position / step.block_size];
      valid = slot == int64_t(page) * step.block_size + position % step.block_size;
    }
    for (int row = 0; row < id; ++row)
      valid &= slot != step.slots[row];
    if (!valid) fail(step.control, 3, 0, call, id, step.health);
  }
  if (pages > 0 && pages <= step.block_table_cols && id < pages) {
    bool live = false;
    for (int row = 0; row < Rows; ++row)
      live |= step.slots[row] >= 0;
    for (int page = id; live && page < pages; page += Groups * Local) {
      const auto block = step.blocks[page];
      if (block < 0 || block >= step.cache_blocks) fail(step.control, 3, 0, call, page, step.health);
    }
  }
}

template <int TP>
inline void store_kv(sycl::nd_item<1> item, DraftWeights weights, Workspace work, Step step) {
  constexpr int Q = Shape::query_heads / TP, KV = Shape::kv_heads / TP;
  constexpr int Qkv = (Q + 2 * KV) * D;
  const auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int task = sg.get_group_linear_id() * Groups + item.get_group_linear_id();
  if (task >= Rows * (Q + KV)) return;
  const int row = task / (Q + KV), head = task % (Q + KV);
  const int64_t slot = step.slots[row];
  if (slot < 0) return;
  const bool is_key = head >= Q;
  auto* qkv = work.qkv + row * Qkv;
  auto* projection = qkv + head * D;
  const auto* norm = is_key ? weights.key_norm : weights.query_norm;
  const auto* rope = step.rope_cache + step.positions[row] * D;
  float norm_sum = 0;
#pragma unroll
  for (int d = lane; d < D; d += 16) {
    const float x = float(projection[d]);
    norm_sum += x * x;
  }
  norm_sum = sycl::reduce_over_group(sg, norm_sum, sycl::plus<float>());
  const float inverse = sycl::rsqrt(norm_sum / D + 1e-6f);
  const int64_t cache_offset = slot / step.block_size * step.cache_block_stride +
                               slot % step.block_size * step.cache_token_stride +
                               sycl::max(0, head - Q) * step.cache_head_stride;
#pragma unroll
  for (int d = lane; d < D / 2; d += 16) {
    const float a = float(scalar_t(float(projection[d]) * inverse * float(norm[d])));
    const float b = float(scalar_t(float(projection[d + D / 2]) * inverse * float(norm[d + D / 2])));
    const float cs = float(rope[d]), sn = float(rope[d + D / 2]);
    const scalar_t first(a * cs - b * sn), second(b * cs + a * sn);
    if (is_key) {
      const auto* value = qkv + (head + KV) * D;
      step.key_cache[cache_offset + d] = first;
      step.key_cache[cache_offset + d + D / 2] = second;
      step.value_cache[cache_offset + d] = value[d];
      step.value_cache[cache_offset + d + D / 2] = value[d + D / 2];
    } else {
      projection[d] = first;
      projection[d + D / 2] = second;
    }
  }
}

template <int TP, bool Attention>
inline bool add_residual_and_reduce(sycl::nd_item<1> item, const scalar_t* local, Step step, uint32_t call) {
  constexpr int PairsPerRow = H / Groups / 2;
  const int lid = item.get_local_linear_id();
  constexpr int Phase = Attention ? 0 : 1;
  float a = 0, b = 0;
  if (lid < Rows * PairsPerRow) {
    a = float(local[lid * 2]);
    b = float(local[lid * 2 + 1]);
  }
  if constexpr (TP == 2) {
    if (!reduce_pairs<Rows>(item, a, b, Phase, call, step)) return false;
  }
  if (lid < Rows * PairsPerRow) {
    const int row = lid / PairsPerRow;
    const int col = (item.get_group_linear_id() * PairsPerRow + lid % PairsPerRow) * 2;
    const int offset = row * H + col;
    if constexpr (Attention) {
      const auto residual = [&](int index) {
        return float(scalar_t(float(step.hidden[index]) + (step.residual ? float(step.residual[index]) : 0.f)));
      };
      step.res_out[offset] = scalar_t(float(scalar_t(a)) + residual(offset));
      step.res_out[offset + 1] = scalar_t(float(scalar_t(b)) + residual(offset + 1));
    } else {
      step.output[offset] = scalar_t(a);
      step.output[offset + 1] = scalar_t(b);
    }
  }
  return true;
}

template <int TP, bool FP8>
void launch_draft(sycl::queue& queue, DraftWeights weights, Workspace work, Step step) {
  static constexpr int QDim = Shape::query_heads / TP * D;
  constexpr int Qkv = (Shape::query_heads + 2 * Shape::kv_heads) / TP * D;
  constexpr int Intermediate = Shape::dense_intermediate / TP;
  queue.submit([&](sycl::handler& cgh) {
    // The same SLM holds packed Q, projection inputs and attention merge state.
    constexpr int ProjectionTiles = Rows * (H > QDim ? H : QDim) / 8;
    constexpr int KvTiles = TP == 1 ? SharedKvTileWords : 0;
    sycl::local_accessor<UInt4, 1> tiles(std::max(ProjectionTiles, KvTiles), cgh);
    sycl::local_accessor<scalar_t, 1> normed(Rows * H, cgh);
    using PartialStorage = std::conditional_t<TP == 2, UInt4, float>;
    constexpr int PartialWords = TP == 2 ? 2 * SharedKvTileWords : 2 * Subgroups * Rows * 16;
    sycl::local_accessor<PartialStorage, 1> projection_partial(
        PartialWords * sizeof(float) / sizeof(PartialStorage), cgh);
    sycl::local_accessor<float, 1> norm_partial(Rows * (Subgroups + 1) + 3, cgh);
    sycl::local_accessor<scalar_t, 1> output_partial(Rows * (H / Groups), cgh);
    sycl::ext::oneapi::experimental::properties properties{sycl::ext::intel::experimental::grf_size<256>};
    cgh.parallel_for<DraftLayer<TP, FP8>>(
        sycl::nd_range<1>(Groups * Local, Local),
        properties,
        [weights, work, step, tiles, normed, projection_partial, norm_partial, output_partial](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(16)]] {
              const uint32_t call = epoch_load(step.control + Epoch) + 1;
              if (Flag(step.control[Failed]).load() || (step.health && Flag(step.health[0]).load())) return;
              check_metadata(item, step, call);
              if (item.get_group_linear_id() == 0 && item.get_local_linear_id() < Rows * (2 / TP) * 2)
                Flag(step.control[Epoch + 16 + item.get_local_linear_id()]).store(0);
              auto* x = normed.get_multi_ptr<sycl::access::decorated::no>().get();
              auto* partial = reinterpret_cast<float*>(
                  projection_partial.template get_multi_ptr<sycl::access::decorated::no>().get());
              auto* norms = norm_partial.get_multi_ptr<sycl::access::decorated::no>().get();
              auto* local = output_partial.get_multi_ptr<sycl::access::decorated::no>().get();
              auto* packed = reinterpret_cast<uint32_t*>(tiles.get_multi_ptr<sycl::access::decorated::no>().get());
              auto* plan = reinterpret_cast<int*>(norms);

              reference_norm<Rows, true, false, NormOutput::PackedProjection>(
                  item, step.hidden, step.residual, weights.input_norm, x, norms, packed);
              wg_bar(item.get_group());
              project<FP8, H, Qkv, TP == 1 ? 3 : 4>(item, weights.qkv, packed, partial, work.qkv);
              if (!grid_barrier_handoff<true>(item, step.control, 1, call, step.health)) return;
              store_kv<TP>(item, weights, work, step);
              if (!grid_barrier_handoff<true>(item, step.control, 2, call, step.health)) return;

              if (step.block_size == 64)
                reference_attention<TP, Rows, false, true, TP == 1, TP == 2>(
                    item, work, step, packed, plan, reinterpret_cast<uint32_t*>(partial));
              else
                reference_attention<TP, Rows, false>(item, work, step, packed, plan);
              if (!grid_barrier_handoff<true>(item, step.control, 3, call, step.health)) return;
              reference_attention_merge<TP, Rows, false>(item, work, step, reinterpret_cast<float*>(packed), plan);
              if (!grid_barrier_handoff<true>(item, step.control, 4, call, step.health)) return;

              pack_projection_input<Rows, QDim>(item, work.attention, QDim, packed);
              wg_bar(item.get_group());
              project<FP8, QDim, H, 8, ProjectionOutput::Local>(item, weights.output, packed, partial, local);
              wg_bar(item.get_group());
              if (!add_residual_and_reduce<TP, true>(item, local, step, call)) return;
              if (!grid_barrier_handoff<true>(item, step.control, 5, call, step.health)) return;

              reference_norm<Rows, false, false, NormOutput::PackedProjection>(
                  item, step.res_out, nullptr, weights.post_norm, x, norms, packed);
              wg_bar(item.get_group());
              auto* intermediate = reinterpret_cast<scalar_t*>(work.gate_up_partial);
              project<FP8, H, Intermediate, TP == 1 ? 2 : 4, ProjectionOutput::SwiGlu>(
                  item, weights.gate_up, packed, partial, intermediate);
              if (!grid_barrier_handoff<true>(item, step.control, 6, call, step.health)) return;

              project<FP8, Intermediate, H, 8, ProjectionOutput::Local>(
                  item, weights.down, reinterpret_cast<uint32_t*>(intermediate), partial, local);
              wg_bar(item.get_group());
              if (!add_residual_and_reduce<TP, false>(item, local, step, call)) return;
              if (item.get_global_linear_id() == 0) Flag(step.control[Epoch]).store(call);
            });
  });
}

}  // namespace

__attribute__((visibility("default"))) void
dispatch(sycl::queue& queue, DraftWeights weights, Workspace work, Step step, int tp) {
  const bool fp8 = weights.qkv.scale.values != nullptr;
  if (tp == 1 && fp8)
    launch_draft<1, true>(queue, weights, work, step);
  else if (tp == 1)
    launch_draft<1, false>(queue, weights, work, step);
  else if (fp8)
    launch_draft<2, true>(queue, weights, work, step);
  else
    launch_draft<2, false>(queue, weights, work, step);
}
}  // namespace sgl_kernel::minicpm5::xe20::b70_draft_bf16
