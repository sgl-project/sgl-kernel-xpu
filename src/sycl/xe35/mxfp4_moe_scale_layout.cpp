/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Relayout of per-token MXFP4 activation scales into the surface the Xe3.5
           MXFP grouped-GEMM mainloop indexes (see moe_grouped_mm_nt_xe35_mxfp4_w4a4).

    `sgl_per_token_group_quant_fp4_xe35` emits scales as a plain row-major [M, G] byte
    surface (one row per token, G = ceil(K/32) groups). The MXFP mainloop instead
    reads, per expert, an [G, round_up(M_e, 4)] tile: M is the *fastest-moving*
    extent ("MN-major"), and its width must be a multiple of 4 because the Xe 2D
    block load of the scales is DWord-granular. This kernel writes that surface:

      out[dst_start_e * G + g * padded_e + i] = in[(src_start_e + i) * G + g]

    with src_start_e the prefix sum of rows_per_expert, padded_e = round_up(rows_e, 4)
    and dst_start_e the prefix sum of the padded counts. Row counts stay on the
    device, so each work group scans them into SLM once rather than syncing them
    to the host, then binary-searches that scan for the expert owning each tile it
    claims.

    The transpose is staged through SLM one 64-row tile at a time so both the loads
    and the stores move contiguous bytes; a direct byte scatter costs ~64x its own
    size in memory traffic. Each work item moves a 32-bit quad on both sides -- four
    scale groups of one row in, four rows of one scale group out -- which needs
    `groups % 4 == 0`; the byte-granular path below covers the rest.
*/

#include "sgl_kernel/xe35/routing_quant.h"

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/Utils.h"

namespace {

constexpr int kScaleAlignRows = 4;
constexpr int kTileRows = 64;
constexpr int kWorkGroupSize = 256;
// Bounds the SLM tile at kTileRows * kMaxGroups bytes (16 KB), i.e. K <= 8192.
constexpr int kMaxGroups = 256;

void check_out_rows_xe35(int64_t out_rows, int64_t in_rows) {
  TORCH_CHECK(
      out_rows >= in_rows,
      "mxfp4_moe_scale_to_mn_major_xe35: out must have at least as many rows as the ",
      in_rows,
      "-row surface it is built from (padding only grows it), got ",
      out_rows);
}

template <bool VEC4>
struct Mxfp4ScaleToMnMajorKernel_xe35 : public __SYCL_KER_CONFIG_CONVENTION__ {
  Mxfp4ScaleToMnMajorKernel_xe35(
      uint8_t* out_,
      const uint8_t* in_,
      const int32_t* rows_per_expert_,
      int32_t num_experts_,
      int32_t groups_,
      int64_t in_rows_,
      int64_t out_rows_,
      int32_t num_work_groups_)
      : out(out_),
        in(in_),
        rows_per_expert(rows_per_expert_),
        num_experts(num_experts_),
        groups(groups_),
        in_rows(in_rows_),
        out_rows(out_rows_),
        num_work_groups(num_work_groups_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    // uint32 elements so the staged tile is 4-byte aligned for the VEC4 path;
    // the byte path just views the same storage as bytes.
    smem = sycl::local_accessor<uint32_t>((kTileRows * groups + 3) / 4, cgh);
    // Per-expert tile/row/dst-row prefixes, scanned once per work group, plus
    // the three work-group-wide partial sums the scan itself runs on.
    prefix = sycl::local_accessor<int32_t>(3 * (num_experts + 1), cgh);
    scan = sycl::local_accessor<int32_t>(3 * kWorkGroupSize, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsz = static_cast<int32_t>(item.get_local_range(0));
    uint8_t* tile = reinterpret_cast<uint8_t*>(&smem[0]);
    int32_t* tile_start = &prefix[0];
    int32_t* src_of = &prefix[num_experts + 1];
    int32_t* dst_of = &prefix[2 * (num_experts + 1)];

    // Scan the (device-resident) row counts into SLM once, so the tile loop below
    // is O(tiles claimed by this work group) instead of a walk over the whole tile
    // list with a modulo per candidate.
    //
    // The scan runs across the whole work group, not on one work item. A serial
    // scan is num_experts *dependent* global loads -- ~50 us for 256 experts,
    // which at decode shapes costs more than moving the scales does. Two-level:
    // each work item folds its own contiguous slice of experts, the per-item
    // sums are scanned in SLM, then each item replays its slice from its
    // exclusive base.
    {
      int32_t* scan_t = &scan[0];
      int32_t* scan_s = &scan[kWorkGroupSize];
      int32_t* scan_d = &scan[2 * kWorkGroupSize];
      const int32_t per_item = (num_experts + lsz - 1) / lsz;
      const int32_t lo_e = sycl::min(lid * per_item, num_experts);
      const int32_t hi_e = sycl::min(lo_e + per_item, num_experts);

      int32_t lt = 0, ls = 0, ld = 0;
      for (int32_t e = lo_e; e < hi_e; ++e) {
        const int32_t rows = rows_per_expert[e];
        lt += (rows + kTileRows - 1) / kTileRows;
        ls += rows;
        ld += (rows + kScaleAlignRows - 1) / kScaleAlignRows * kScaleAlignRows;
      }
      scan_t[lid] = lt;
      scan_s[lid] = ls;
      scan_d[lid] = ld;
      item.barrier(sycl::access::fence_space::local_space);

      // Hillis-Steele inclusive scan over the lsz per-item sums.
      for (int32_t off = 1; off < lsz; off <<= 1) {
        int32_t at = 0, as = 0, ad = 0;
        if (lid >= off) {
          at = scan_t[lid - off];
          as = scan_s[lid - off];
          ad = scan_d[lid - off];
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (lid >= off) {
          scan_t[lid] += at;
          scan_s[lid] += as;
          scan_d[lid] += ad;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      int32_t bt = lid > 0 ? scan_t[lid - 1] : 0;
      int32_t bs = lid > 0 ? scan_s[lid - 1] : 0;
      int32_t bd = lid > 0 ? scan_d[lid - 1] : 0;
      for (int32_t e = lo_e; e < hi_e; ++e) {
        const int32_t rows = rows_per_expert[e];
        tile_start[e] = bt;
        src_of[e] = bs;
        dst_of[e] = bd;
        bt += (rows + kTileRows - 1) / kTileRows;
        bs += rows;
        bd += (rows + kScaleAlignRows - 1) / kScaleAlignRows * kScaleAlignRows;
      }
      // Exactly one work item owns the last expert, so exactly one writes the
      // totals into the one-past-the-end slots.
      if (lo_e < num_experts && hi_e == num_experts) {
        tile_start[num_experts] = bt;
        src_of[num_experts] = bs;
        dst_of[num_experts] = bd;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int32_t total_tiles = tile_start[num_experts];
    for (int32_t t = wg; t < total_tiles; t += num_work_groups) {
      // Which expert owns tile t: tile_start is non-decreasing, so an upper-bound
      // search over num_experts entries replaces the old linear re-walk.
      int32_t lo = 0;
      int32_t hi = num_experts - 1;
      while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= t) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      const int32_t e = lo;
      const int32_t rows = rows_per_expert[e];
      const int32_t padded = (rows + kScaleAlignRows - 1) / kScaleAlignRows * kScaleAlignRows;
      const int64_t src_start = src_of[e];
      const int64_t dst_start = dst_of[e];
      const int32_t row0 = (t - tile_start[e]) * kTileRows;
      {
        const int32_t tile_rows = sycl::min(kTileRows, rows - row0);

        if constexpr (VEC4) {
          // Round the tile out to the 4-row scale alignment so the store side
          // can emit whole quads. `row0` is a multiple of kTileRows and the last
          // tile of an expert ends at round_up(rows, 4) == padded, so the extra
          // rows stay inside this expert's own padded block -- rows the GEMM
          // pairs with pad rows of A and whose output is dropped on combine.
          const int32_t tile_rows4 = (tile_rows + kScaleAlignRows - 1) / kScaleAlignRows * kScaleAlignRows;
          const int32_t quads_per_row = groups / 4;

          // Load: consecutive work items read consecutive quads of a source row.
          for (int32_t idx = lid; idx < tile_rows4 * quads_per_row; idx += lsz) {
            const int32_t r = idx / quads_per_row;
            const int32_t q = idx - r * quads_per_row;
            const int64_t row = src_start + row0 + r;
            uint32_t v = 0;
            if (r < tile_rows && row < in_rows) {
              v = *reinterpret_cast<const uint32_t*>(in + row * groups + q * 4);
            }
            *reinterpret_cast<uint32_t*>(tile + r * groups + q * 4) = v;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Store: consecutive work items write consecutive row quads of one
          // scale group, so the 32-bit stores stay contiguous within a group.
          const int32_t row_quads = tile_rows4 / 4;
          for (int32_t idx = lid; idx < groups * row_quads; idx += lsz) {
            const int32_t g = idx / row_quads;
            const int32_t rq = idx - g * row_quads;
            const uint8_t* src = tile + rq * 4 * groups + g;
            const uint32_t packed = static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[groups]) << 8) |
                                    (static_cast<uint32_t>(src[2 * groups]) << 16) |
                                    (static_cast<uint32_t>(src[3 * groups]) << 24);
            const int64_t off = dst_start * groups + static_cast<int64_t>(g) * padded + row0 + rq * 4;
            if (off + 4 <= out_rows * groups) {
              *reinterpret_cast<uint32_t*>(out + off) = packed;
            }
          }
        } else {
          const int32_t count = tile_rows * groups;

          // Load: consecutive work items read consecutive bytes of a source row.
          for (int32_t idx = lid; idx < count; idx += lsz) {
            const int64_t row = src_start + row0 + idx / groups;
            tile[idx] = row < in_rows ? in[row * groups + idx % groups] : uint8_t(0);
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Store: consecutive work items write consecutive rows of one scale group.
          for (int32_t idx = lid; idx < count; idx += lsz) {
            const int32_t g = idx / tile_rows;
            const int32_t i = idx % tile_rows;
            const int64_t off = dst_start * groups + static_cast<int64_t>(g) * padded + row0 + i;
            if (off < out_rows * groups) {
              out[off] = tile[i * groups + g];
            }
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
    }
  }

  uint8_t* out;
  const uint8_t* in;
  const int32_t* rows_per_expert;
  int32_t num_experts;
  int32_t groups;
  int64_t in_rows;
  int64_t out_rows;
  int32_t num_work_groups;
  sycl::local_accessor<uint32_t> smem;
  sycl::local_accessor<int32_t> prefix;
  sycl::local_accessor<int32_t> scan;
};

}  // namespace

void mxfp4_moe_scale_to_mn_major_xe35(torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert) {
  TORCH_CHECK(
      in.device().is_xpu() && out.device().is_xpu() && rows_per_expert.device().is_xpu(),
      "mxfp4_moe_scale_to_mn_major_xe35: all tensors must be on an XPU device");
  TORCH_CHECK(in.dim() == 2 && out.dim() == 2, "mxfp4_moe_scale_to_mn_major_xe35: in/out must be 2D");
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "mxfp4_moe_scale_to_mn_major_xe35: in/out must be contiguous");
  TORCH_CHECK(in.size(1) == out.size(1), "mxfp4_moe_scale_to_mn_major_xe35: in/out must have the same group count");
  TORCH_CHECK(
      in.size(1) > 0 && in.size(1) <= kMaxGroups,
      "mxfp4_moe_scale_to_mn_major_xe35: group count must be in [1, ",
      kMaxGroups,
      "], got ",
      in.size(1));
  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.scalar_type() == at::kInt,
      "mxfp4_moe_scale_to_mn_major_xe35: rows_per_expert must be a 1D int32 tensor");
  TORCH_CHECK(rows_per_expert.is_contiguous(), "mxfp4_moe_scale_to_mn_major_xe35: rows_per_expert must be contiguous");
  check_out_rows_xe35(out.size(0), in.size(0));
  // e8m0 scales are carried either as uint8 or as the typed view.
  auto is_e8m0 = [](const torch::Tensor& t) {
    return t.scalar_type() == at::kByte || t.scalar_type() == at::kFloat8_e8m0fnu;
  };
  TORCH_CHECK(
      is_e8m0(in) && is_e8m0(out),
      "mxfp4_moe_scale_to_mn_major_xe35: scales must be uint8 or float8_e8m0fnu, got ",
      in.scalar_type(),
      " and ",
      out.scalar_type());

  const int32_t num_experts = static_cast<int32_t>(rows_per_expert.numel());
  const int32_t groups = static_cast<int32_t>(in.size(1));
  if (num_experts == 0 || in.size(0) == 0) {
    return;
  }

  // One work group per tile would need the (device-side) row counts on the host;
  // instead a fixed grid strides over the tile list, sized to cover the buffer's
  // worth of tiles without launching more work groups than there is work for.
  // Every work group pays one num_experts-long scan, so do not oversubscribe.
  const int64_t max_tiles = (in.size(0) + kTileRows - 1) / kTileRows + num_experts;
  const int32_t num_work_groups = static_cast<int32_t>(std::min<int64_t>(max_tiles, 1024));

  auto queue = at::xpu::getCurrentXPUStream(in.device().index()).queue();

#define LAUNCH_SCALE_RELAYOUT(VEC4)                                                                           \
  do {                                                                                                        \
    Mxfp4ScaleToMnMajorKernel_xe35<VEC4> kernel(                                                                   \
        reinterpret_cast<uint8_t*>(out.data_ptr()),                                                           \
        reinterpret_cast<const uint8_t*>(in.data_ptr()),                                                      \
        rows_per_expert.data_ptr<int32_t>(),                                                                  \
        num_experts,                                                                                          \
        groups,                                                                                               \
        in.size(0),                                                                                           \
        out.size(0),                                                                                          \
        num_work_groups);                                                                                     \
    sycl_kernel_submit(static_cast<size_t>(num_work_groups) * kWorkGroupSize, kWorkGroupSize, queue, kernel); \
  } while (0)

  // A row of scales must be a whole number of 32-bit quads for the vectorized
  // path; K a multiple of 128 (groups a multiple of 4) covers every shape this
  // MoE runs. Everything else falls back to the byte-granular walk.
  if (groups % 4 == 0) {
    LAUNCH_SCALE_RELAYOUT(true);
  } else {
    LAUNCH_SCALE_RELAYOUT(false);
  }
#undef LAUNCH_SCALE_RELAYOUT
}
