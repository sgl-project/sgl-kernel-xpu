/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"

namespace cutlass::fmha::kernel {

// Scatter a large appended KV range once before attention. The fused mainloop
// path remains preferable for short appends because it avoids a second launch.
template <class ElementK_, class ElementV_, int NumSGs_ = 16>
class XeFMHAAppendKVKernel {
 public:
  using ElementK = ElementK_;
  using ElementV = ElementV_;
  static constexpr int NumSGs = NumSGs_;
  static constexpr int SgSize = cute::intel::sg_size;
  static constexpr int ThreadsPerWG = NumSGs * SgSize;
  static constexpr int TokensPerWG = NumSGs;

  struct Arguments {
    ElementK* ptr_K_cache = nullptr;
    ElementV* ptr_V_cache = nullptr;
    ElementK const* ptr_K_new = nullptr;
    ElementV const* ptr_V_new = nullptr;
    int const* ptr_cu_seqlens_k_new = nullptr;
    int seq_len_kv_new = 0;
    int const* ptr_cache_seqlens = nullptr;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int max_num_pages_per_seq = 0;
    int batch = 0;
    int num_heads_kv = 0;
    int head_size_qk = 0;
    int head_size_vo = 0;
    int max_seq_len_kv_new = 0;
    bool const* skip_batch_mask = nullptr;
  };

  using Params = Arguments;
  struct SharedStorage {};
  static constexpr int SharedStorageSize = 0;

  static Params to_underlying_arguments(Arguments const& args, void*) {
    return args;
  }

  static bool can_implement(Arguments const& args) {
    return args.ptr_K_cache != nullptr && args.ptr_V_cache != nullptr &&
           args.ptr_K_new != nullptr && args.ptr_V_new != nullptr &&
           args.ptr_cache_seqlens != nullptr && args.ptr_page_table != nullptr &&
           args.batch > 0 && args.num_heads_kv > 0 && args.head_size_qk > 0 &&
           args.head_size_vo > 0 && args.max_seq_len_kv_new > 0 &&
           (args.ptr_cu_seqlens_k_new != nullptr || args.seq_len_kv_new > 0) &&
           args.page_size > 0 && args.max_num_pages_per_seq > 0;
  }

  static int get_workspace_size(Arguments const&) {
    return 0;
  }

  static cutlass::Status initialize_workspace(
      Arguments const&, void* = nullptr, void* = nullptr, void* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(
        uint32_t(cute::ceil_div(params.max_seq_len_kv_new, TokensPerWG)),
        uint32_t(params.num_heads_kv),
        uint32_t(params.batch));
  }

  static dim3 get_block_shape() {
    return dim3(ThreadsPerWG, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char*) const {
    int const batch_idx = int(BlockIdxZ());
    if (params.skip_batch_mask != nullptr && params.skip_batch_mask[batch_idx]) return;

    int new_begin;
    int new_len;
    if (params.ptr_cu_seqlens_k_new != nullptr) {
      new_begin = params.ptr_cu_seqlens_k_new[batch_idx];
      new_len = params.ptr_cu_seqlens_k_new[batch_idx + 1] - new_begin;
    } else {
      new_len = params.seq_len_kv_new;
      new_begin = batch_idx * new_len;
    }

    int const token = int(BlockIdxX()) * TokensPerWG + int(ThreadIdxX()) / SgSize;
    if (token >= new_len) return;

    int const lane = int(ThreadIdxX()) % SgSize;
    int const kv_head = int(BlockIdxY());
    int const dst_token = params.ptr_cache_seqlens[batch_idx] + token;
    int const page = dst_token / params.page_size;
    int const page_token = dst_token - page * params.page_size;
    int const dst_row =
        params.ptr_page_table[batch_idx * params.max_num_pages_per_seq + page] *
            params.page_size +
        page_token;
    int const src_token = new_begin + token;

    size_t const k_src =
        (size_t(src_token) * params.num_heads_kv + kv_head) * params.head_size_qk;
    size_t const v_src =
        (size_t(src_token) * params.num_heads_kv + kv_head) * params.head_size_vo;
    size_t const k_dst =
        (size_t(dst_row) * params.num_heads_kv + kv_head) * params.head_size_qk;
    size_t const v_dst =
        (size_t(dst_row) * params.num_heads_kv + kv_head) * params.head_size_vo;

    copy_row(params.ptr_K_cache, params.ptr_K_new, k_dst, k_src, params.head_size_qk, lane);
    copy_row(params.ptr_V_cache, params.ptr_V_new, v_dst, v_src, params.head_size_vo, lane);
  }

 private:
  template <class Element>
  CUTLASS_DEVICE static void copy_row(
      Element* dst, Element const* src, size_t dst_offset, size_t src_offset, int elements, int lane) {
    using Vec32 = cute::intel::uint8;
    using Vec16 = cutlass::ulonglong2;
    if constexpr (sizeof(Element) == 2) {
      if ((elements % 16) == 0) {
        for (int i = lane; i < elements / 16; i += SgSize) {
          int const offset = i * 16;
          *reinterpret_cast<Vec32*>(dst + dst_offset + offset) =
              *reinterpret_cast<Vec32 const*>(src + src_offset + offset);
        }
        return;
      }
      if ((elements % 8) == 0) {
        for (int i = lane; i < elements / 8; i += SgSize) {
          int const offset = i * 8;
          *reinterpret_cast<Vec16*>(dst + dst_offset + offset) =
              *reinterpret_cast<Vec16 const*>(src + src_offset + offset);
        }
        return;
      }
    }
    for (int d = lane; d < elements; d += SgSize) {
      dst[dst_offset + d] = src[src_offset + d];
    }
  }
};

}  // namespace cutlass::fmha::kernel
