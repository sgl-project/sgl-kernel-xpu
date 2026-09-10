#pragma once

// Solve-A kernel: fuses the compute-A step (K x K^T, causal-masked and
// beta/decay-scaled) with the lower-triangular matrix-inverse step into a
// single kernel launch.
//
// Design:
//   1) Compute K x K^T with a 64x64x32 TiledMMA, and apply the causal-mask/beta/decay
//      scaling to obtain the masked A tile in registers, then write it to global-memory.
//   2) Invert the four 16x16 diagonal blocks directly against global
//      memory, using a forward-substitution/register-broadcast algorithm.
//   3) Subgroup 0 computes all off-diagonal blocks sequentially with
//      intermediate dependencies in registers and no barriers.

#include "chunk_gated_delta_rule_kernels.hpp"

namespace gdn {
using namespace cute;

// Each workgroup is responsible for one (chunk, v_head) pair.
// TiledMMAComputeA must be a 4-subgroup TiledMMA (chunk_gemm_policy_compute_A).
// TiledMMAInverse must be the 1-subgroup, 16x16x16 TiledMMA (chunk_gemm_policy_inverse).
template <typename T, class TiledMMAComputeA, class TiledMMAInverse>
CUTE_DEVICE void chunk_compute_A_inverse_fused_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    const T* k,
    const float* b,
    const float* a,
    const int* total_chunks,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  const int flat_chunk_id = item.get_group(0);
  const int v_head_id = item.get_group(1);

  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_local_id = sg.get_local_linear_id();

  if (flat_chunk_id >= *total_chunks) {
    return;
  }

  float* slm_mem = static_cast<float*>(slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());
  float* g_slm_ptr = slm_mem;

  const int kv_ratio = num_v_heads / num_k_heads;
  const int chunk_start_offset = flat_chunk_id * chunk_size;

  // load a to slm. a.shape = [v_head_num, total_virtual_seqlen]
  CUTE_UNROLL
  for (int e = local_id; e < chunk_size; e += local_range) {
    g_slm_ptr[e] = a[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
  }

  item.barrier(sycl::access::fence_space::local_space);

  // ---------------------------------------------------------------------
  // Phase 1: A = mask(K x K^T) * beta, using TiledMMAComputeA
  // ---------------------------------------------------------------------
  TiledMMAComputeA mmaA{};
  auto wg_tile = mmaA.tile_mnk();
  auto thr_mma = mmaA.get_slice(local_id);

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M = get<1>(typename TiledMMAComputeA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMAComputeA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int m_tile_start = 0;
  int n_tile_start = 0;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  auto k_ptr =
      k + static_cast<int64_t>(chunk_start_offset) * num_k_heads * head_k_dim + (v_head_id / kv_ratio) * head_k_dim;
  auto K_tensor_shape = make_shape(chunk_size, head_k_dim);
  auto K_tensor =
      make_tensor(make_gmem_ptr(k_ptr), make_layout(K_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

  auto A_ptr =
      A + static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size + chunk_start_offset * chunk_size;
  auto A_tensor_shape = make_shape(chunk_size, chunk_size);
  auto A_tensor = make_tensor(make_gmem_ptr(A_ptr), make_layout(A_tensor_shape, make_stride(chunk_size, _1{})));

  Tensor cA_id = make_identity_tensor(A_tensor_shape);
  Tensor gA_C = local_tile(cA_id, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

  auto copy_A_c = get_block_2d_copy_D<void>(mmaA, A_tensor);
  auto thr_copy_A_c = copy_A_c.get_slice(local_id);
  auto tCrA_c = thr_copy_A_c.partition_sg_fragment_S(gA_C);
  auto tCgA_c = thr_copy_A_c.partition_D(gA_C);
  auto tSrA_c = thr_mma.partition_sg_fragment_C(gA_C);

  clear(tSrA_c);
  gemm_TTS(K_tensor, K_tensor, tSrA_c, 0, 0, mmaA);

  CUTE_UNROLL
  for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
    int n_idx = n_tile_start + n_sg_start + sn * sub_group_size + sg_local_id;
    CUTE_UNROLL
    for (int sm = 0; sm < SG_M; ++sm) {
      int m_idx = m_tile_start + m_sg_start + sm;
      float beta_value = b[(chunk_start_offset + m_idx) + v_head_id * total_virtual_seqlen];

      tSrA_c(sn * SG_M + sm) *= sycl::exp(g_slm_ptr[m_idx] - g_slm_ptr[n_idx]) * beta_value;
      if (m_idx == n_idx) {
        tSrA_c(sn * SG_M + sm) = 1.0f;
      }
      if (m_idx < n_idx) {
        tSrA_c(sn * SG_M + sm) = 0.0f;
      }
    }
  }

  reorder(tSrA_c, tCrA_c);
  copy(copy_A_c, tCrA_c, tCgA_c);

  // Ensure that the A tile is visible in global memory before Phase 2 below reads it back.
  item.barrier(sycl::access::fence_space::global_and_local);

  // ---------------------------------------------------------------------
  // Phase 2: invert the four 16x16 diagonal blocks directly in global
  // memory,  each subgroup inverts one diagonal block in parallel.
  // ---------------------------------------------------------------------
  {
    int i = sg_id;  // requires sg_range == 4 (chunk_gemm_policy_compute_A)
    int offset = i * A_block_size;
    T* A_ptr_xx = A_ptr + offset * chunk_size + offset;
    float A_local[A_block_size];
    float A_other[A_block_size];
    float A_sum;
    CUTE_UNROLL
    for (int e = 0; e < sg_local_id + 1; ++e) {
      A_local[e] = 0.0f;
    }

    T A_load[A_block_size];
    CUTE_UNROLL
    for (int e = 0; e < sg_local_id; ++e) {
      A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
    }

    CUTE_UNROLL
    for (int mm_idx = 1; mm_idx < A_block_size; ++mm_idx) {
      CUTE_UNROLL
      for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
        float send_value = static_cast<float>(A_load[nn_idx]);
        float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
        if (sg_local_id == nn_idx) {
          A_local[mm_idx] = receive_value;
        }
      }
    }

    CUTE_UNROLL
    for (int mm_idx = 1; mm_idx < A_block_size; ++mm_idx) {
      A_sum = 0.0f;
      CUTE_UNROLL
      for (int e = 1; e < mm_idx + 1; ++e) {
        A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
      }

      CUTE_UNROLL
      for (int e = 1; e < mm_idx + 1; ++e) {
        A_sum += A_local[e] * A_other[e];
      }

      A_local[mm_idx] = -A_local[mm_idx] - A_sum;
    }

    CUTE_UNROLL
    for (int e = sg_local_id + 1; e < A_block_size; ++e) {
      A_ptr_xx[e * chunk_size + sg_local_id] = static_cast<T>(A_local[e]);
    }
  }

  // Ensure that diagonal blocks are visible in global memory before Phase 3 below reads it back.
  item.barrier(sycl::access::fence_space::global_and_local);

  // ---------------------------------------------------------------------
  // Phase 3: subgroup 0 computes the off-diagonal blocks sequentially,
  // keeping each dependency chain's intermediate results resident in
  // registers. Only subgroup 0 enters this scope, so the other three
  // subgroups leave the workgroup-barrier sequence once their diagonal
  // block (Phase 2) is done, without deadlocking the workgroup.
  // ---------------------------------------------------------------------
  if (sg_id == 0) {
    int local_id = sg_local_id;
    using TiledMMA = TiledMMAInverse;
    TiledMMA mma{};
    auto wg_tile = mma.tile_mnk();
    auto thr_mma = mma.get_slice(local_id);

    auto A_ptr_11 = A_ptr;

    auto A_ptr_21 = A_ptr + A_block_size * chunk_size;
    auto A_ptr_22 = A_ptr + A_block_size * chunk_size + A_block_size;

    auto A_ptr_31 = A_ptr + 2 * A_block_size * chunk_size;
    auto A_ptr_32 = A_ptr + 2 * A_block_size * chunk_size + A_block_size;
    auto A_ptr_33 = A_ptr + 2 * A_block_size * chunk_size + 2 * A_block_size;

    auto A_ptr_41 = A_ptr + 3 * A_block_size * chunk_size;
    auto A_ptr_42 = A_ptr + 3 * A_block_size * chunk_size + A_block_size;
    auto A_ptr_43 = A_ptr + 3 * A_block_size * chunk_size + 2 * A_block_size;
    auto A_ptr_44 = A_ptr + 3 * A_block_size * chunk_size + 3 * A_block_size;

    auto A_XX_tensor_shape = make_shape(A_block_size, A_block_size);

    auto A_11_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_11), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    auto A_21_tensor =
        make_tensor(make_gmem_ptr(A_ptr_21), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_21_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_21), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_22_tensor =
        make_tensor(make_gmem_ptr(A_ptr_22), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_22_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_22), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    auto A_31_tensor =
        make_tensor(make_gmem_ptr(A_ptr_31), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_31_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_31), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_32_tensor =
        make_tensor(make_gmem_ptr(A_ptr_32), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_32_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_32), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_33_tensor =
        make_tensor(make_gmem_ptr(A_ptr_33), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_33_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_33), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    auto A_41_tensor =
        make_tensor(make_gmem_ptr(A_ptr_41), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_41_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_41), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_42_tensor =
        make_tensor(make_gmem_ptr(A_ptr_42), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_42_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_42), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_43_tensor =
        make_tensor(make_gmem_ptr(A_ptr_43), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto A_43_tensor_T =
        make_tensor(make_gmem_ptr(A_ptr_43), make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto A_44_tensor =
        make_tensor(make_gmem_ptr(A_ptr_44), make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

    Tensor cA = make_identity_tensor(A_XX_tensor_shape);
    Tensor cB = make_identity_tensor(A_XX_tensor_shape);
    Tensor cC = make_identity_tensor(A_XX_tensor_shape);
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(0, _));
    Tensor gC = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrC = thr_mma.partition_sg_fragment_C(gC);

    // Every off-diagonal GEMM below has exactly one K-tile (K=A_block_size=16, matching
    // TiledMMAInverse's own K), so the GEMM helpers' mainloop barrier --
    // whose only purpose is keeping subgroups aligned across successive
    // K-tile iterations -- is a provable no-op here and is skipped
    // entirely (gemm_T*S's `NoBarrier` template argument).
    auto phase3_gemm_tts = [&](auto const& lhs, auto const& rhs, auto& accum) {
      gemm_TTS<ScopeSubgroup, true>(lhs, rhs, accum, 0, 0, mma);
    };
    auto phase3_gemm_sts = [&](auto const& lhs_reg, auto const& rhs, auto& accum) {
      gemm_STS<ScopeSubgroup, true>(lhs_reg, rhs, accum, 0, 0, mma);
    };
    // `gemm_TSS` reads its A operand from global memory (a diagonal
    // inverse) and its B operand from an already-computed register
    // fragment (an off-diagonal inverse kept resident via make_transposed_b_fragment).
    auto phase3_gemm_tss = [&](auto const& lhs, auto const& rhs_reg, auto& accum) {
      gemm_TSS<ScopeSubgroup, true>(lhs, rhs_reg, accum, 0, 0, mma);
    };

    // Column 1: inv21 -> inv31 -> inv41. Each finished (or partial) result
    // is reordered directly into a register B fragment and fed to the next
    // GEMM via phase3_gemm_tss, instead of round-tripping through global
    // memory and a transposed reload.
    auto copy_D_21 = get_block_2d_copy_D<void>(mma, A_21_tensor);
    auto thr_copy_D_21 = copy_D_21.get_slice(local_id);
    auto tCrD_21 = thr_copy_D_21.partition_sg_fragment_S(gC);
    auto tCgD_21 = thr_copy_D_21.partition_D(gC);
    auto inv21_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    clear(tCrC);
    phase3_gemm_tts(A_22_tensor, A_21_tensor_T, tCrC);
    reorder(tCrC, tCrA);
    clear(tCrC);
    phase3_gemm_sts(tCrA, A_11_tensor_T, tCrC);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_21);
    copy(copy_D_21, tCrD_21, tCgD_21);
    {
      auto inv21_b_view = make_transposed_b_fragment(mma, inv21_b);
      reorder(tCrC, inv21_b_view);
    }

    auto copy_D_31 = get_block_2d_copy_D<void>(mma, A_31_tensor);
    auto thr_copy_D_31 = copy_D_31.get_slice(local_id);
    auto tCrD_31 = thr_copy_D_31.partition_sg_fragment_S(gC);
    auto tCgD_31 = thr_copy_D_31.partition_D(gC);
    auto inv31_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    clear(tCrC);
    phase3_gemm_tts(A_31_tensor, A_11_tensor_T, tCrC);
    phase3_gemm_tss(A_32_tensor, inv21_b, tCrC);
    {
      auto partial31_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      auto partial31_b_view = make_transposed_b_fragment(mma, partial31_b);
      reorder(tCrC, partial31_b_view);
      clear(tCrC);
      phase3_gemm_tss(A_33_tensor, partial31_b, tCrC);
    }
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_31);
    copy(copy_D_31, tCrD_31, tCgD_31);
    {
      auto inv31_b_view = make_transposed_b_fragment(mma, inv31_b);
      reorder(tCrC, inv31_b_view);
    }

    auto copy_D_41 = get_block_2d_copy_D<void>(mma, A_41_tensor);
    auto thr_copy_D_41 = copy_D_41.get_slice(local_id);
    auto tCrD_41 = thr_copy_D_41.partition_sg_fragment_S(gC);
    auto tCgD_41 = thr_copy_D_41.partition_D(gC);
    clear(tCrC);
    phase3_gemm_tts(A_41_tensor, A_11_tensor_T, tCrC);
    phase3_gemm_tss(A_42_tensor, inv21_b, tCrC);
    phase3_gemm_tss(A_43_tensor, inv31_b, tCrC);
    {
      auto partial41_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      auto partial41_b_view = make_transposed_b_fragment(mma, partial41_b);
      reorder(tCrC, partial41_b_view);
      clear(tCrC);
      phase3_gemm_tss(A_44_tensor, partial41_b, tCrC);
    }
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_41);
    copy(copy_D_41, tCrD_41, tCgD_41);

    // Column 2: inv32 -> inv42. Same register-resident pattern as column 1.
    auto copy_D_32 = get_block_2d_copy_D<void>(mma, A_32_tensor);
    auto thr_copy_D_32 = copy_D_32.get_slice(local_id);
    auto tCrD_32 = thr_copy_D_32.partition_sg_fragment_S(gC);
    auto tCgD_32 = thr_copy_D_32.partition_D(gC);
    auto inv32_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    clear(tCrC);
    phase3_gemm_tts(A_33_tensor, A_32_tensor_T, tCrC);
    reorder(tCrC, tCrA);
    clear(tCrC);
    phase3_gemm_sts(tCrA, A_22_tensor_T, tCrC);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_32);
    copy(copy_D_32, tCrD_32, tCgD_32);
    {
      auto inv32_b_view = make_transposed_b_fragment(mma, inv32_b);
      reorder(tCrC, inv32_b_view);
    }

    auto copy_D_42 = get_block_2d_copy_D<void>(mma, A_42_tensor);
    auto thr_copy_D_42 = copy_D_42.get_slice(local_id);
    auto tCrD_42 = thr_copy_D_42.partition_sg_fragment_S(gC);
    auto tCgD_42 = thr_copy_D_42.partition_D(gC);
    clear(tCrC);
    phase3_gemm_tts(A_42_tensor, A_22_tensor_T, tCrC);
    phase3_gemm_tss(A_43_tensor, inv32_b, tCrC);
    {
      auto partial42_b = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      auto partial42_b_view = make_transposed_b_fragment(mma, partial42_b);
      reorder(tCrC, partial42_b_view);
      clear(tCrC);
      phase3_gemm_tss(A_44_tensor, partial42_b, tCrC);
    }
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_42);
    copy(copy_D_42, tCrD_42, tCgD_42);

    // Column 3: inv43. No downstream chain, so nothing to keep resident.
    auto copy_D_43 = get_block_2d_copy_D<void>(mma, A_43_tensor);
    auto thr_copy_D_43 = copy_D_43.get_slice(local_id);
    auto tCrD_43 = thr_copy_D_43.partition_sg_fragment_S(gC);
    auto tCgD_43 = thr_copy_D_43.partition_D(gC);
    clear(tCrC);
    phase3_gemm_tts(A_44_tensor, A_43_tensor_T, tCrC);
    reorder(tCrC, tCrA);
    clear(tCrC);
    phase3_gemm_sts(tCrA, A_33_tensor_T, tCrC);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= -1.0f;
    }
    reorder(tCrC, tCrD_43);
    copy(copy_D_43, tCrD_43, tCgD_43);
  }
}

template <typename T, typename StateTag>
class ChunkComputeAInverseFusedKernel;

// Launcher: submits a single kernel that replaces the ChunkComputeAKernel +
// ChunkInverseOptKernel pair from the earlier two-kernel design (see
// kernel_launcher in chunk_gated_delta_rule_kernels.hpp, which now
// calls this instead). `total_chunks` and the cumulative-sum decay values
// in `a` must already have been produced by chunk_prepare_kernel before
// this is launched.
template <typename T, typename StateT>
void launch_chunk_compute_A_inverse_fused(
    sycl::queue& queue,
    T* A,
    const T* k,
    const float* b,
    const float* a,
    int* total_chunks,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  TORCH_CHECK(is_bmg(), "chunk_gdn: only BMG is supported for now");
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<cute::detail::subgroup_size>, intelex::grf_size<256>};

  using WGTileComputeA = chunk_gemm_policy_compute_A::WGTile;
  using SGLayoutComputeA = chunk_gemm_policy_compute_A::SGLayout;
  using MMAComputeA =
      typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTileComputeA>, SGLayoutComputeA>::TiledMMA;

  using WGTileInverse = chunk_gemm_policy_inverse::WGTile;
  using SGLayoutInverse = chunk_gemm_policy_inverse::SGLayout;
  // Off-diagonal blocks are read/written directly in global memory (see
  // Phase 3 in chunk_compute_A_inverse_fused_kernel), using the same T
  // (bf16) DPAS op as MMAComputeA -- exactly like chunk_inverse_opt_kernel's
  // own MMAInverse.
  using MMAInverse = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTileInverse>, SGLayoutInverse>::TiledMMA;

  auto mmaComputeA = MMAComputeA{};
  int MaxThreadsPerWorkgroupComputeA = size(mmaComputeA);
  sycl::range<3> local_compute_A(1, 1, MaxThreadsPerWorkgroupComputeA);

  // Grid is over-provisioned since the real chunk count depends on the
  // ragged per-batch sequence lengths, which is only known on device (same
  // as chunk_compute_A_kernel's launch config).
  const int total_chunks_upper_bound = div_up(total_virtual_seqlen, chunk_size);
  sycl::range<3> global_compute_A(total_chunks_upper_bound, num_v_heads, 1);

  // g cache in SLM, exactly like chunk_compute_A_kernel's own slm_mem_const.
  int slm_size_compute_A = chunk_size;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(sycl::range<1>(slm_size_compute_A), cgh);
    cgh.parallel_for<ChunkComputeAInverseFusedKernel<T, StateT>>(
        sycl::nd_range<3>{global_compute_A * local_compute_A, local_compute_A}, kernel_props, [=](auto) {
          chunk_compute_A_inverse_fused_kernel<T, MMAComputeA, MMAInverse>(
              local_mem,
              A,
              k,
              b,
              a,
              total_chunks,
              total_virtual_seqlen,
              num_k_heads,
              head_k_dim,
              num_v_heads,
              head_v_dim);
        });
  });
}

}  // namespace gdn
