// SPDX-License-Identifier: Apache-2.0
#pragma once

// Adapted from mono_decode/xe_2/mono_layers.hpp MONO_ATTN_STEP and
// mono_layers_b.hpp attn_layer_b, revision
// 922432bc4288f7a323cd550b2599fb318d4568a9. Included inside
// sgl_kernel::minicpm5::xe20::mono after the model pointer types.

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL UInt8 __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, Int2 xy);
extern "C" SYCL_EXTERNAL UInt8 __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, Int2 xy);
#endif

constexpr int DynamicAttentionTokens = 96 * 1024;

inline int attention_partial_groups(int tp) {
#ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
  return Groups * Subgroups / (MaxRows * (2 / tp) * 2);
#else
  return Groups;
#endif
}

template <int TP, int Rows, bool SharedSequence = false>
struct AttentionChunks {
  static constexpr int Slots = Groups * Subgroups / ((2 / TP) * 2);
  // Preserve the padded bucket's attention reduction boundaries for real
  // requests, while excluding dummy requests from every execution stage.
  static constexpr int PlannerRows = Rows <= 2 ? Rows : (Rows <= 4 ? 4 : 8);
  int width, first[Rows + 1];
  bool dynamic;

  static int sequence_length(const Step& step, int row) {
    return step.lengths[SharedSequence ? 0 : row];
  }

  explicit AttentionChunks(const Step& step) {
    int total = 0;
#pragma unroll
    for (int r = 0; r < Rows; ++r)
      total += step.slots[r] < 0 ? 0 : sequence_length(step, r);
    width = sycl::max(16, ((total + Slots - PlannerRows - 1) / (Slots - PlannerRows) + 15) & ~15);
    first[0] = 0;
    // M1/M2 keep static slots; longer rows increase the chunk width.
    dynamic = Rows > 2 && total >= DynamicAttentionTokens;
    if (dynamic) {
      int nonempty = 0;
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        nonempty += step.slots[r] >= 0 && sequence_length(step, r) > 0;
      int64_t cumulative = 0;
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        const int length = step.slots[r] < 0 ? 0 : sequence_length(step, r);
        cumulative += length;
        nonempty -= length > 0;
        int end = int((cumulative * Slots + total / 2) / total);
        end = sycl::max(end, first[r] + (length > 0 ? 1 : 0));
        first[r + 1] = sycl::min(end, Slots - nonempty);
      }
      return;
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      const int length = step.slots[r] < 0 ? 0 : sequence_length(step, r);
      first[r + 1] = first[r] + (length + width - 1) / width;
    }
  }
};

// The plan is identical for every subgroup of a workgroup. Keep its
// runtime-indexed prefix array in SLM, as the reference's shared staging,
// instead of allocating a private array in every work-item.
struct SharedAttentionChunks {
  int width;
  const int* first;
  bool dynamic;

  explicit SharedAttentionChunks(const int* plan) : width(plan[0]), first(plan + 2), dynamic(plan[1] != 0) {}
};

inline UInt8 attention_k_tile(const scalar_t* row, int k, int pitch) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
      intptr_t(row), D * 2 - 1, 15, pitch * 2 - 1, Int2{k / 2, 0});
#else
  return UInt8{};
#endif
}

inline UInt8 attention_v_tile(const scalar_t* row, int d, int pitch) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
      intptr_t(row), D * 2 - 1, 15, pitch * 2 - 1, Int2{d, 0});
#else
  return UInt8{};
#endif
}

constexpr int SharedKvTileWords = 8 * 4 * 16 * 8;

namespace shared_kv_io = sycl::ext::oneapi::experimental;
constexpr auto SharedKvBlockProperties = shared_kv_io::properties{
    shared_kv_io::data_placement_striped,
    shared_kv_io::contiguous_memory,
    shared_kv_io::full_group,
    shared_kv_io::alignment<16>,
    shared_kv_io::detail::native_local_block_io};

// One subgroup writes a striped block, keeping its DPAS operand order.
inline void reference_stage_shared_kv(sycl::nd_item<1> item, const Step& step, uint32_t* staging, int start) {
  constexpr int TilesPerHead = 4 * (D / 16);
  const auto sg = item.get_sub_group();
#pragma unroll
  for (int task = sg.get_group_linear_id(); task < 2 * TilesPerHead; task += Subgroups) {
    const int head = task / TilesPerHead, tile = task % TilesPerHead;
    const int token_tile = tile / (D / 16), dim_tile = tile % (D / 16);
    const int position = start + token_tile * 16;
    const int64_t base = int64_t(step.blocks[position / 64]) * step.cache_block_stride + head * step.cache_head_stride +
                         (position % 64) * step.cache_token_stride;
    const auto* row = step.key_cache + base;
    const auto key = attention_k_tile(row, dim_tile * 16, step.cache_token_stride);
    const auto value = attention_v_tile(step.value_cache + base, dim_tile * 16, step.cache_token_stride);
    const int offset = (dim_tile * 4 + token_tile) * 16 * 8;
    auto* keys = staging + head * 2 * SharedKvTileWords + offset;
    const auto local =
        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(keys);
    shared_kv_io::group_store(
        sg, sycl::bit_cast<sycl::vec<uint32_t, 8>>(key), local.get_decorated(), SharedKvBlockProperties);
    shared_kv_io::group_store(
        sg,
        sycl::bit_cast<sycl::vec<uint32_t, 8>>(value),
        local.get_decorated() + SharedKvTileWords,
        SharedKvBlockProperties);
  }
}

inline UInt8
reference_read_shared_tile(sycl::sub_group sg, const uint32_t* staging, int head, int tile, int dim_tile, bool value) {
  const auto* words = staging + (head * 2 + int(value)) * SharedKvTileWords + (dim_tile * 4 + tile) * 16 * 8;
  const auto local =
      sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(words);
  sycl::vec<uint32_t, 8> result;
  shared_kv_io::group_load(sg, local.get_decorated(), result, SharedKvBlockProperties);
  return sycl::bit_cast<UInt8>(result);
}

// Each FP32 component occupies one SIMD16 GRF. Converting it into one half
// of the destination GRF creates 16-bit [head][token] rows in DPAS A order.
inline UInt4 attention_pack_probabilities(Float8 probability) {
  UInt4 output{};
#ifdef __SYCL_DEVICE_ONLY__
  asm("{\n"
      ".decl P v_type=G type=f num_elts=128 alias=<%1,0>\n"
#ifdef MINICPM5_MONO_FP16
      ".decl A v_type=G type=hf num_elts=128 alias=<%0,0>\n"
#else
      ".decl A v_type=G type=bf num_elts=128 alias=<%0,0>\n"
#endif
      "mov (M1, 16) A(0,0)<1> P(0,0)<16;16,1>\n"
      "mov (M1, 16) A(0,16)<1> P(1,0)<16;16,1>\n"
      "mov (M1, 16) A(1,0)<1> P(2,0)<16;16,1>\n"
      "mov (M1, 16) A(1,16)<1> P(3,0)<16;16,1>\n"
      "mov (M1, 16) A(2,0)<1> P(4,0)<16;16,1>\n"
      "mov (M1, 16) A(2,16)<1> P(5,0)<16;16,1>\n"
      "mov (M1, 16) A(3,0)<1> P(6,0)<16;16,1>\n"
      "mov (M1, 16) A(3,16)<1> P(7,0)<16;16,1>\n"
      "}\n"
      : "=rw"(output)
      : "rw"(probability));
#endif
  return output;
}

// Stage one operand at a time in dead norm storage, keeping packed Q separate.
template <int TP, bool Values>
inline void stage_verify_operand(sycl::nd_item<1> item, const Step& step, uint32_t* storage, int start, int width) {
  constexpr int TilesPerHead = 4 * (D / 16);
  const auto sg = item.get_sub_group();
#pragma unroll
  for (int task = sg.get_group_linear_id(); task < 2 * TilesPerHead; task += Subgroups) {
    const int bank = task / TilesPerHead, tile = task % TilesPerHead;
    const int token_tile = tile / (D / 16), dim_tile = tile % (D / 16);
    const int head = TP == 1 ? bank : 0;
    int position = start + token_tile * 16;
    if constexpr (TP == 2) {
      constexpr int SubgroupsPerChunk = Shape::max_rows * 2;
      const int own_bank = sg.get_group_linear_id() / SubgroupsPerChunk;
      position += (bank - own_bank) * width;
    }
    const int64_t base = int64_t(step.blocks[position / 64]) * step.cache_block_stride + head * step.cache_head_stride +
                         (position % 64) * step.cache_token_stride;
    const UInt8 operand = Values ? attention_v_tile(step.value_cache + base, dim_tile * 16, step.cache_token_stride)
                                 : attention_k_tile(step.key_cache + base, dim_tile * 16, step.cache_token_stride);
    auto* words = storage + bank * SharedKvTileWords + (dim_tile * 4 + token_tile) * 16 * 8;
    const auto local =
        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(words);
    shared_kv_io::group_store(
        sg, sycl::bit_cast<sycl::vec<uint32_t, 8>>(operand), local.get_decorated(), SharedKvBlockProperties);
  }
}

template <
    int TP,
    int Rows,
    bool Gated = true,
    bool FixedPages = false,
    bool VerifyProbabilityPack = false,
    bool ShareVerifyKv = false>
inline void reference_attention(
    sycl::nd_item<1> item, Workspace work, Step step, uint32_t* packed, int* plan, uint32_t* verify_kv = nullptr) {
  constexpr bool DirectProbabilityPack = FixedPages && (std::is_same_v<scalar_t, sycl::half> || VerifyProbabilityPack);
  constexpr int Q = 32 / TP, KV = 2 / TP;
  constexpr int QStride = (Gated ? 2 : 1) * D, QKV = Q * QStride + 2 * KV * D;
  constexpr int Words = (D / 16) * 16 * 4;
  constexpr bool SharedDraft = TP == 1 && Rows == Shape::draft_rows && FixedPages && !Gated;
  constexpr bool SharedTp2Draft = ShareVerifyKv && TP == 2 && Rows == Shape::draft_rows && FixedPages && !Gated;
  constexpr bool CanGroupByChunk = SharedDraft || ShareVerifyKv || (TP == 1 && Rows == 8 && FixedPages && Gated);
  static_assert(!ShareVerifyKv || SharedTp2Draft || ((TP == 1 || TP == 2) && Rows == 8 && FixedPages && Gated));
  constexpr bool CanStageKv = SharedDraft || ShareVerifyKv;
  constexpr bool StaticVerify =
      Gated && TP == 2 && Rows == 8 && FixedPages && std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>;
  using Chunks = AttentionChunks<TP, Rows, !Gated>;
  const int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  if (lid == 0) {
    const Chunks chunks(step);
    bool shared_draft = false;
    if constexpr (SharedDraft || SharedTp2Draft) {
      const int length = step.lengths[0];
      shared_draft = step.block_table_stride == 0 && length >= (DynamicAttentionTokens + Rows - 1) / Rows;
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        shared_draft &= step.slots[r] >= 0;
    }
    plan[0] = chunks.width;
    plan[1] = chunks.dynamic || shared_draft;
#pragma unroll
    for (int r = 0; r <= Rows; ++r)
      plan[2 + r] = shared_draft ? r * Groups * (SharedTp2Draft ? 2 : 1) : chunks.first[r];
  }
  for (int index = lid; index < Rows * KV * 2 * Words; index += Local) {
    const int group = index / Words, word = index % 4;
    const int l = index / 4 % 16, k = index / 64 % (D / 16);
    const int r = group / (KV * 2), head = group % (KV * 2) * 8;
    const int offset = r * QKV + (head + operand_row(l, word)) * QStride + k * 16 + operand_column(l);
    packed[index] = pack16(work.qkv[offset], work.qkv[offset + 1]);
  }
  wg_bar(item.get_group());
  const SharedAttentionChunks chunks(plan);
  const int subgroup = sg.get_group_linear_id();
  const int wg = item.get_group_linear_id();
  constexpr int ChunksPerGroup = TP == 2 && ShareVerifyKv ? 2 : 1;
  constexpr int ChunksPerRow = Groups * ChunksPerGroup;
  bool group_by_chunk = false, stage_kv = false;
  int common_length = 0;
  if constexpr (CanGroupByChunk) {
    static_assert(Subgroups >= Rows * KV * 2);
    group_by_chunk = chunks.dynamic && step.block_table_stride == 0;
#pragma unroll
    for (int r = 1; r <= Rows; ++r)
      group_by_chunk &= chunks.first[r] == r * ChunksPerRow;
    if constexpr (CanStageKv) {
      stage_kv = group_by_chunk;
      common_length = step.lengths[0];
      if constexpr (ShareVerifyKv) {
        const int denominator = ChunksPerRow * 16;
        const int width = (common_length + denominator - 1) / denominator;
#pragma unroll
        for (int r = 1; r < Rows; ++r) {
          const int length = step.lengths[r];
          stage_kv &= (length + denominator - 1) / denominator == width;
          common_length = sycl::min(common_length, length);
        }
        if constexpr (TP == 2) group_by_chunk = stage_kv;
      }
    }
  }
  // A TP2 WG covers two adjacent chunks for each row, one per subgroup cohort.
  // Retain the original logical partial indices when assigning chunk owners.
  // The spare draft subgroups help stage KV and follow identical barriers.
  // Their duplicate query work has no partial output.
  constexpr int CohortSubgroups = ChunksPerGroup == 2 ? Subgroups / 2 : Subgroups;
  const int cohort = subgroup / CohortSubgroups, within = subgroup % CohortSubgroups;
  const bool duplicate = (SharedDraft || SharedTp2Draft) && stage_kv && within >= Rows * KV * 2;
  const int row_head = duplicate ? within % (KV * 2) : within;
  const int chunk = wg * ChunksPerGroup + cohort;
  int job = subgroup * Groups + wg;
  if (group_by_chunk) {
    const int row = row_head / (KV * 2);
    const int first = SharedTp2Draft ? chunks.first[row] : row * ChunksPerRow;
    job = (first + chunk) * (KV * 2) + row_head % (KV * 2);
  }
  const int slot = job / (KV * 2), kvh = job / 2 % KV;
  if (slot < chunks.first[Rows]) {
    int r = 0;
#pragma unroll
    for (int b = 1; b < Rows; ++b)
      if (chunks.first[b] <= slot) r = b;
    const int head = kvh * 16 + (job % 2) * 8;
    const uint32_t* qpk = packed + (r * KV * 2 + kvh * 2 + job % 2) * Words;
    UInt4 query_registers[SharedDraft ? D / 16 : 1];
    if constexpr (SharedDraft) {
      if (stage_kv) {
#pragma unroll
        for (int k = 0; k < D / 16; ++k)
          query_registers[k] = *reinterpret_cast<const UInt4*>(qpk + (k * 16 + lane) * 4);
        // All Q readers finish before the packed region becomes K/V storage.
        wg_bar(item.get_group());
      }
    }
    const auto query_at = [&](int k) {
      if constexpr (SharedDraft)
        if (stage_kv) return query_registers[k];
      return *reinterpret_cast<const UInt4*>(qpk + (k * 16 + lane) * 4);
    };
    const int chunk_count = chunks.first[r + 1] - chunks.first[r];
    const int length = Chunks::sequence_length(step, r);
    const int chunk_width =
        Rows > 2 && chunks.dynamic ? ((length + chunk_count * 16 - 1) / (chunk_count * 16)) * 16 : chunks.width;
    auto process_chunk = [&](int logical_slot) {
      Float8 maximum, normalizer{}, values[8]{};
#pragma unroll
      for (int h = 0; h < 8; ++h)
        maximum[h] = -INFINITY;

      // NT=4 is only called when the complete 64-token tile fits.
      auto tile_step = [&]<int NT, bool SharedTile = false>(int start, int end) {
        constexpr bool ReuseVerifySlm = SharedTile && ShareVerifyKv;
        int64_t offsets[NT];
        Float8 scores[NT]{};
        if constexpr (SharedTile) {
          static_assert(CanStageKv && NT == 4);
          if constexpr (ReuseVerifySlm)
            stage_verify_operand<TP, false>(item, step, verify_kv, start, chunk_width);
          else
            reference_stage_shared_kv(item, step, packed, start);
          wg_bar(item.get_group());
        }
        if constexpr (FixedPages && NT == 4) {
          const int TokenStride = step.cache_token_stride;
          if constexpr (!SharedTile) {
            constexpr int Page = 64;
            const int* table = step.blocks + r * step.block_table_stride;
            const int first_page = start / Page, offset = start % Page;
            const auto cache_row = [&](int page, int token) {
              return int64_t(table[page]) * step.cache_block_stride + kvh * step.cache_head_stride +
                     token * TokenStride;
            };
            const auto first = cache_row(first_page, 0);
            const auto last = offset == 0 ? first : cache_row(first_page + 1, 0);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const int token = offset + t * 16;
              offsets[t] = (token < Page ? first : last) + (token % Page) * TokenStride;
            }
          }
          // Each pair of score tiles uses the same Q operand. The K16
          // accumulation order within each tile remains unchanged.
#pragma unroll
          for (int t = 0; t < NT; t += 2)
#pragma unroll
            for (int k = 0; k < D; k += 16) {
              const UInt4 query = query_at(k / 16);
              UInt8 k0, k1;
              if constexpr (SharedTile) {
                const int bank = TP == 1 ? kvh : cohort;
                const auto* stage = ReuseVerifySlm ? verify_kv + bank * SharedKvTileWords : packed;
                const int source_head = ReuseVerifySlm ? 0 : kvh;
                k0 = reference_read_shared_tile(sg, stage, source_head, t, k / 16, false);
                k1 = reference_read_shared_tile(sg, stage, source_head, t + 1, k / 16, false);
              } else {
                k0 = attention_k_tile(step.key_cache + offsets[t], k, TokenStride);
                k1 = attention_k_tile(step.key_cache + offsets[t + 1], k, TokenStride);
              }
              dpas16(scores[t], query, k0);
              dpas16(scores[t + 1], query, k1);
            }
        } else {
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const int pos = start + t * 16;
            const int block = step.blocks[r * step.block_table_stride + pos / step.block_size];
            offsets[t] = int64_t(block) * step.cache_block_stride + kvh * step.cache_head_stride +
                         (pos % step.block_size) * step.cache_token_stride;
#pragma unroll
            for (int k = 0; k < D; k += 16) {
              UInt8 keys;
              if constexpr (NT == 4) {
                keys = attention_k_tile(step.key_cache + offsets[t], k, step.cache_token_stride);
              } else if (pos + 16 <= end) {
                keys = attention_k_tile(step.key_cache + offsets[t], k, step.cache_token_stride);
              } else {
#pragma unroll
                for (int p = 0; p < 8; ++p)
                  keys[p] = pos + lane < end
                                ? pack16(
                                      step.key_cache[offsets[t] + lane * step.cache_token_stride + k + 2 * p],
                                      step.key_cache[offsets[t] + lane * step.cache_token_stride + k + 2 * p + 1])
                                : 0u;
              }
              dpas16(scores[t], query_at(k / 16), keys);
            }
          }
        }
        if constexpr (ReuseVerifySlm) {
          // All K readers are done. Stage V during the softmax work.
          wg_bar(item.get_group());
          stage_verify_operand<TP, true>(item, step, verify_kv, start, chunk_width);
        }
        // Keep separate score scaling/subtraction rounding, including NT4.
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int h = 0; h < 8; ++h)
            scores[t][h] = start + t * 16 + lane < end ? scores[t][h] * 0.12751743082459868f : -INFINITY;
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          float local = scores[0][h];
#pragma unroll
          for (int t = 1; t < NT; ++t)
            local = sycl::max(local, scores[t][h]);
          const float next = sycl::max(maximum[h], sycl::reduce_over_group(sg, local, sycl::maximum<float>()));
          const float alpha = next == -INFINITY ? 1.f : sycl::native::exp2(maximum[h] - next);
          maximum[h] = next;
          normalizer[h] *= alpha;
#pragma unroll
          for (int d = 0; d < 8; ++d)
            values[d][h] *= alpha;
        }
        if constexpr (ReuseVerifySlm) wg_bar(item.get_group());
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          UInt4 pa;
          if constexpr (DirectProbabilityPack) {
            Float8 probability;
#pragma unroll
            for (int h = 0; h < 8; ++h) {
              const float p = maximum[h] == -INFINITY ? 0.f : sycl::native::exp2(scores[t][h] - maximum[h]);
              normalizer[h] += p;
              probability[h] = p;
            }
            pa = attention_pack_probabilities(probability);
          } else {
            UInt4 probability{};
#pragma unroll
            for (int h = 0; h < 8; ++h) {
              const float p = maximum[h] == -INFINITY ? 0.f : sycl::native::exp2(scores[t][h] - maximum[h]);
              normalizer[h] += p;
              probability[h / 2] |= uint32_t(sycl::bit_cast<uint16_t>(scalar_t(p))) << (16 * (h % 2));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int token = operand_column(lane);
              // Two head probabilities travel in one dword. Preserve the
              // original scalar_t bits with half as many subgroup shuffles.
              const uint32_t a = sycl::select_from_group(sg, probability[j], token);
              const uint32_t b = sycl::select_from_group(sg, probability[j], token + 1);
              const int shift = lane < 8 ? 0 : 16;
              pa[j] = ((a >> shift) & 0xffffu) | ((b >> shift) << 16);
            }
          }
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            UInt8 value;
            if constexpr (SharedTile) {
              if constexpr (ReuseVerifySlm)
                value = reference_read_shared_tile(
                    sg, verify_kv + (TP == 1 ? kvh : cohort) * SharedKvTileWords, 0, t, d, false);
              else
                value = reference_read_shared_tile(sg, packed, kvh, t, d, true);
            } else if constexpr (NT == 4) {
              value = attention_v_tile(step.value_cache + offsets[t], d * 16, step.cache_token_stride);
            } else if (start + t * 16 + 16 <= end) {
              value = attention_v_tile(step.value_cache + offsets[t], d * 16, step.cache_token_stride);
            } else {
#pragma unroll
              for (int p = 0; p < 8; ++p) {
                const int position = start + t * 16 + 2 * p;
                const scalar_t a = position < end
                                       ? step.value_cache[offsets[t] + 2 * p * step.cache_token_stride + d * 16 + lane]
                                       : scalar_t(0.f);
                const scalar_t b =
                    position + 1 < end
                        ? step.value_cache[offsets[t] + (2 * p + 1) * step.cache_token_stride + d * 16 + lane]
                        : scalar_t(0.f);
                value[p] = pack16(a, b);
              }
            }
            dpas16(values[d], pa, value);
          }
        }
        if constexpr (SharedTile) wg_bar(item.get_group());
      };
      // Each logical chunk owns its partial slot. Dynamic workers may claim
      // chunks in any order, but the merge's grouping/order stays
      // deterministic.
      const int begin = (logical_slot - chunks.first[r]) * chunk_width;
      const int end = sycl::min(begin + chunk_width, length);
      int tile = begin;
      if constexpr (CanStageKv) {
        if (stage_kv) {
          // The common part contains complete 64-token tiles for every
          // causal row. Keep each row's original path for its own tail.
          // Both cohorts must execute the same number of WG barriers.
          constexpr int LastCohort = ChunksPerGroup - 1;
          const int last_start = (wg * ChunksPerGroup + LastCohort) * chunk_width;
          const int common_end = begin + sycl::min(chunk_width, common_length - last_start);
          for (; tile + 64 <= common_end; tile += 64)
            tile_step.template operator()<4, true>(tile, end);
        }
      }
      for (; tile + 64 <= end; tile += 64)
        tile_step.template operator()<4>(tile, end);
      for (; tile < end; tile += 16)
        tile_step.template operator()<1>(tile, end);
      if constexpr (SharedDraft || SharedTp2Draft)
        if (duplicate) return;
#pragma unroll
      for (int h = 0; h < 8; ++h) {
        const size_t offset = (size_t(logical_slot) * Q + head + h) * (D + 2);
#pragma unroll
        for (int d = 0; d < 8; ++d)
          work.attention_partial[offset + d * 16 + lane] = values[d][h];
        const float sum = sycl::reduce_over_group(sg, normalizer[h], sycl::plus<float>());
        if (lane == 0) {
          work.attention_partial[offset + D] = maximum[h];
          work.attention_partial[offset + D + 1] = sum;
        }
      }
    };
    if constexpr (Rows <= 2 || StaticVerify) {
      process_chunk(slot);
    } else {
      const int ticket = (r * KV + kvh) * 2 + job % 2;
      // Keep one attention body for both schedules. Reload the immutable
      // WG-local mode inside the loop so LLVM cannot unswitch/duplicate the
      // large QK/PV state for a second static path.
      const volatile int* mode = plan + 1;
      for (;;) {
        const bool dynamic = *mode != 0 && !group_by_chunk;
        int logical_slot = slot;
        if (dynamic) {
          uint32_t claimed = 0;
          if (lane == 0) claimed = Flag(step.control[Epoch + 16 + ticket]).fetch_add(1);
          claimed = sycl::select_from_group(sg, claimed, 0);
          if (claimed >= uint32_t(chunk_count)) break;
          logical_slot = chunks.first[r] + int(claimed);
        }
        process_chunk(logical_slot);
        if (!dynamic) break;
      }
    }
  }
}

template <int TP, int Rows, bool Gated = true, bool ParallelFixedPageMerge = false>
inline void reference_attention_merge(sycl::nd_item<1> item, Workspace work, Step step, float* local, const int* plan) {
  constexpr int Q = 32 / TP, QKV = 2 * Q * D + 2 * (2 / TP) * D;
  using Chunks = AttentionChunks<TP, Rows, !Gated>;
  constexpr int Slots = Chunks::Slots;
  constexpr bool ParallelOrdinaryWidths = Gated && Groups == B70::work_groups &&
                                          std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16> &&
                                          (Rows == 3 || (Rows >= 5 && Rows <= 7));
  const SharedAttentionChunks chunks(plan);
  const int lid = item.get_local_linear_id();
  for (int job = item.get_group_linear_id(); job < Q * 2; job += Groups) {
    const int h = job / 2, dim = job % 2 * 64;
    if constexpr (
        Rows == 2 || Rows == 4 || (Rows == Shape::draft_rows && !Gated) || Rows == 8 || ParallelOrdinaryWidths) {
      bool parallel_merge = true;
      if constexpr (Rows == Shape::draft_rows && !Gated) {
        // Draft rows share one history. For long histories distribute each
        // row's partition statistics across the work-group.
        parallel_merge = Chunks::sequence_length(step, 0) >= 8192;
      }
      if constexpr (ParallelOrdinaryWidths) {
        bool long_partition = false;
#pragma unroll
        for (int r = 0; r < Rows; ++r)
          long_partition |= step.lengths[r] >= 8192 && chunks.first[r + 1] - chunks.first[r] >= 32;
        parallel_merge = step.block_table_stride != 0 && long_partition;
      }
      if constexpr (Rows == 8) {
        // Real pad8 profiles have64--68 chunks per live row at n5--7,
        // versus32--34 at n8. Keep the measured short-chunk path unchanged.
        parallel_merge = false;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#ifdef MINICPM5_MONO_FP16
          if constexpr (TP == 1)
            parallel_merge |= step.lengths[r] >= 8192 && chunks.first[r + 1] - chunks.first[r] >= 24;
          else
#endif
            parallel_merge |= ParallelFixedPageMerge
                                  ? step.lengths[r] >= 8192 && chunks.first[r + 1] - chunks.first[r] >= 24
                                  : chunks.first[r + 1] - chunks.first[r] >= 48;
        }
      }
      if (parallel_merge) {
        // mono_layers_b.hpp attn_layer_b merge: one work-item per chunk for
        // statistics, then WGS/Rows work-items per row and eight independent
        // numerator loads. Mini stores unnormalized chunk numerators, so keep
        // p and p*chunk_denominator separately instead of using PR's one wgt.
        constexpr int MergeRows = Rows <= 2 ? Rows : Rows <= 4 ? 4 : 8;
        constexpr int T = Local / MergeRows, Parts = T / 64;
        static_assert(Slots <= Local && T % 64 == 0);
        // The original packed-Q storage is dead after attention body and
        // larger than M2's old serial merge workspace. No SLM is added.
        static_assert(2 * Slots + Rows + Subgroups + Local <= Rows * (2 / TP) * 2 * (D / 16) * 16 * 4);
        auto* mkey = reinterpret_cast<int*>(local + 2 * Slots);
        auto* lwp = local + 2 * Slots + Rows;
        auto* red = lwp + Subgroups;
        const auto key = [](float f) {
          const int k = sycl::bit_cast<int>(f);
          return k >= 0 ? k : k ^ 0x7fffffff;
        };
        const auto unkey = [](int k) { return sycl::bit_cast<float>(k >= 0 ? k : k ^ 0x7fffffff); };
        if (lid < Rows) mkey[lid] = key(-INFINITY);
        wg_bar(item.get_group());
        const int g = lid;
        int gb = 0;
#pragma unroll
        for (int r = 1; r < Rows; ++r)
          if (chunks.first[r] <= g) gb = r;
        const bool live = g < chunks.first[Rows];
        float mg = -INFINITY, lg = 0.f;
        if (live) {
          const size_t offset = (size_t(g) * Q + h) * (D + 2);
          mg = work.attention_partial[offset + D];
          lg = work.attention_partial[offset + D + 1];
        }
        if (live && lg > 0.f) {
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>
              maximum(mkey[gb]);
          maximum.fetch_max(key(mg));
        }
        wg_bar(item.get_group());
        if (live) {
          const float p = lg > 0.f ? sycl::native::exp2(mg - unkey(mkey[gb])) : 0.f;
          local[g] = p;
          local[Slots + g] = p * lg;
        }
        wg_bar(item.get_group());
        const auto sg = item.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int b = lid / T, r = lid % T, part = r / 64, d = r % 64;
        // The rounded-up lanes take part in WG barriers. Clamp their
        // metadata index and give them no partitions.
        const int row = MergeRows == Rows ? b : sycl::min(b, Rows - 1);
        const int g0 = chunks.first[row];
        const int n = MergeRows != Rows && b >= Rows ? 0 : chunks.first[row + 1] - g0;
        float lw = 0.f;
        for (int i = r; i < n; i += T)
          lw += local[Slots + g0 + i];
        lw = sycl::reduce_over_group(sg, lw, sycl::plus<float>());
        if (lane == 0) lwp[lid / 16] = lw;
        float a8[8]{};
        int i = part;
        for (; i + 7 * Parts < n; i += 8 * Parts) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int slot = g0 + i + q * Parts;
            a8[q] += local[slot] * work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + dim + d];
          }
        }
        for (; i < n; i += Parts) {
          const int slot = g0 + i;
          a8[0] += local[slot] * work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + dim + d];
        }
        float acc = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q)
          acc += a8[q];
        red[lid] = acc;
        wg_bar(item.get_group());
        if (lid < Rows * 64) {
          const int b2 = lid / 64, d2 = lid % 64;
          float denominator = 0.f, numerator = 0.f;
#pragma unroll
          for (int s = 0; s < T / 16; ++s)
            denominator += lwp[b2 * (T / 16) + s];
#pragma unroll
          for (int p = 0; p < Parts; ++p)
            numerator += red[(b2 * Parts + p) * 64 + d2];
          const scalar_t value = scalar_t(numerator * (denominator > 0.f ? 1.f / denominator : 0.f));
          auto& out = work.attention[(b2 * Q + h) * D + dim + d2];
          if constexpr (Gated) {
            const float gate = float(work.qkv[b2 * QKV + h * 2 * D + D + dim + d2]);
            out = scalar_t(float(value) * float(scalar_t(1.f / (1.f + sycl::exp(-gate)))));
          } else {
            out = value;
          }
        }
        wg_bar(item.get_group());
        continue;
      }
    }
    if constexpr (Rows == 1) {
      // Gather strided partition statistics in parallel, then retain the
      // original ascending-slot denominator and numerator accumulation.
      constexpr int MaximumScratch = 2 * Slots + 1;
      const auto sg = item.get_sub_group();
      const int lane = sg.get_local_linear_id();
      const int subgroup = sg.get_group_linear_id();
      float maximum = -INFINITY;
      for (int slot = chunks.first[0] + lid; slot < chunks.first[1]; slot += Local)
        maximum = sycl::max(maximum, work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + D]);
      maximum = sycl::reduce_over_group(sg, maximum, sycl::maximum<float>());
      if (lane == 0) local[MaximumScratch + subgroup] = maximum;
      wg_bar(item.get_group());
      if (lid == 0) {
        maximum = -INFINITY;
#pragma unroll
        for (int s = 0; s < Subgroups; ++s)
          maximum = sycl::max(maximum, local[MaximumScratch + s]);
        local[MaximumScratch] = maximum;
      }
      wg_bar(item.get_group());
      maximum = local[MaximumScratch];
      for (int slot = chunks.first[0] + lid; slot < chunks.first[1]; slot += Local) {
        const size_t offset = (size_t(slot) * Q + h) * (D + 2);
        local[slot] = maximum == -INFINITY ? 0.f : sycl::native::exp2(work.attention_partial[offset + D] - maximum);
        local[Slots + slot] = work.attention_partial[offset + D + 1];
      }
      wg_bar(item.get_group());
      if (lid == 0) {
        float denominator = 0.f;
        int slot = chunks.first[0];
#ifdef MINICPM5_MONO_FP16
        constexpr bool LoadDenominatorsInBlocks = true;
#else
        constexpr bool LoadDenominatorsInBlocks = Gated && Groups == B70::work_groups;
#endif
        if constexpr (LoadDenominatorsInBlocks) {
          for (; slot + 7 < chunks.first[1]; slot += 8) {
            // Keep the ascending addition chain after loading coefficients.
            Float8 coefficients, denominators;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
              coefficients[p] = local[slot + p];
              denominators[p] = local[Slots + slot + p];
            }
#pragma unroll
            for (int p = 0; p < 8; ++p)
              denominator += coefficients[p] * denominators[p];
          }
        }
        for (; slot < chunks.first[1]; ++slot)
          denominator += local[slot] * local[Slots + slot];
        local[2 * Slots] = denominator > 0.f ? 1.f / denominator : 0.f;
      }
    } else {
      if (lid < Rows) {
        const int r = lid;
        float maximum = -INFINITY, denominator = 0.f;
        for (int slot = chunks.first[r]; slot < chunks.first[r + 1]; ++slot)
          maximum = sycl::max(maximum, work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + D]);
        for (int slot = chunks.first[r]; slot < chunks.first[r + 1]; ++slot) {
          const size_t offset = (size_t(slot) * Q + h) * (D + 2);
          const float p = maximum == -INFINITY ? 0.f : sycl::native::exp2(work.attention_partial[offset + D] - maximum);
          local[r * Slots + slot] = p;
          denominator += p * work.attention_partial[offset + D + 1];
        }
        local[Rows * Slots + r] = denominator > 0.f ? 1.f / denominator : 0.f;
      }
    }
    wg_bar(item.get_group());
    if constexpr (Rows == 1) {
      // As in the reference batched merge, split chunks across the entire
      // WG and keep eight independent reads/accumulations in flight.
      constexpr int Parts = Local / 64;
      static_assert(Parts == 8 && 2 * Slots >= Local);
      const int part = lid / 64, d = dim + lid % 64;
      float sums[8]{};
      int slot = chunks.first[0] + part;
      for (; slot + 7 * Parts < chunks.first[1]; slot += 8 * Parts) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const int current = slot + p * Parts;
          sums[p] += local[current] * work.attention_partial[(size_t(current) * Q + h) * (D + 2) + d];
        }
      }
      for (; slot < chunks.first[1]; slot += Parts)
        sums[0] += local[slot] * work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + d];
      float numerator = 0.f;
#pragma unroll
      for (int p = 0; p < 8; ++p)
        numerator += sums[p];
      // All coefficient reads finish before their SLM is reused. For TP1,
      // the old denominator array is also dead; the inverse stays at or above
      // 512.
      wg_bar(item.get_group());
      local[lid] = numerator;
      wg_bar(item.get_group());
    }
    for (int index = lid; index < Rows * 64; index += Local) {
      const int r = index / 64, d = dim + index % 64;
      float numerator = 0.f;
      if constexpr (Rows == 1) {
#pragma unroll
        for (int part = 0; part < Local / 64; ++part)
          numerator += local[part * 64 + index];
      } else {
        for (int slot = chunks.first[r]; slot < chunks.first[r + 1]; ++slot)
          numerator += local[r * Slots + slot] * work.attention_partial[(size_t(slot) * Q + h) * (D + 2) + d];
      }
      constexpr int Inverse = Rows == 1 ? 2 * Slots : Rows * Slots;
      const scalar_t value = scalar_t(numerator * local[Inverse + r]);
      auto& out = work.attention[(r * Q + h) * D + d];
      if constexpr (Gated) {
        const float gate = float(work.qkv[r * QKV + h * 2 * D + D + d]);
        out = scalar_t(float(value) * float(scalar_t(1.f / (1.f + sycl::exp(-gate)))));
      } else {
        out = value;
      }
    }
    wg_bar(item.get_group());
  }
}
