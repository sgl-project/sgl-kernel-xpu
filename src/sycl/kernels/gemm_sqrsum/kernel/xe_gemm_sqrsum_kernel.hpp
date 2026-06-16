#pragma once

#include "../collective/xe_gemm_sqrsum_mainloop.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/compat/dims.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"

namespace cutlass::gemm_sqrsum::kernel {
using namespace cute;

template <class CollectiveMainloop_>
class GemmSqrSumKernel {
 public:
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementC = typename CollectiveMainloop::ElementC;
  using ElementSqrSum = typename CollectiveMainloop::ElementSqrSum;

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  struct Params {
    typename CollectiveMainloop::Params mainloop;
    cutlass::KernelHardwareInfo hw_info;

    int M;
    int K;
    int N;

    int split_k;

    ElementA const* ptr_A;
    int64_t stride_A_m;
    int64_t stride_A_k;

    ElementB const* ptr_B;
    int64_t stride_B_k;
    int64_t stride_B_n;

    ElementC* ptr_C;
    int64_t stride_C_m;
    int64_t stride_C_n;

    ElementSqrSum* ptr_sqrsum;

    ElementSqrSum* ptr_sqrsum_scratch;
    int64_t stride_sqsc_m;
    int64_t stride_sqsc_n;
  };

  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop;

    int M;
    int K;
    int N;

    int split_k;

    ElementA const* ptr_A;
    int64_t stride_A_m;
    int64_t stride_A_k;

    ElementB const* ptr_B;
    int64_t stride_B_k;
    int64_t stride_B_n;

    ElementC* ptr_C;
    int64_t stride_C_m;
    int64_t stride_C_n;

    ElementSqrSum* ptr_sqrsum;

    ElementSqrSum* ptr_sqrsum_scratch;
    int64_t stride_sqsc_m;
    int64_t stride_sqsc_n;
  };

  struct SharedStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        cutlass::KernelHardwareInfo{},
        args.M,
        args.K,
        args.N,
        args.split_k,
        args.ptr_A,
        args.stride_A_m,
        args.stride_A_k,
        args.ptr_B,
        args.stride_B_k,
        args.stride_B_n,
        args.ptr_C,
        args.stride_C_m,
        args.stride_C_n,
        args.ptr_sqrsum,
        args.ptr_sqrsum_scratch,
        args.stride_sqsc_m,
        args.stride_sqsc_n};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop);
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  static compat::dim3 get_grid_shape(Params const& params) {
    int grid_m = (params.M + BLK_M - 1) / BLK_M;
    int grid_n = (params.N + BLK_N - 1) / BLK_N;
    return compat::dim3(params.split_k, grid_n, grid_m);
  }

  static compat::dim3 get_block_shape() {
    constexpr int num_threads = cute::size(typename CollectiveMainloop::TiledMMA{});
    return compat::dim3(num_threads, 1, 1);
  }

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto item = this_work_item::get_nd_item<3>();
    int thr_id = int(item.get_local_id(2));
    int blk_m = int(item.get_group(0));
    int blk_n = int(item.get_group(1));
    int split_idx = int(item.get_group(2));

    int k_tiles_total = (params.K + BLK_K - 1) / BLK_K;
    int split_k = params.split_k;
    int tiles_per_split = (k_tiles_total + split_k - 1) / split_k;
    int k_tile_begin = split_idx * tiles_per_split;
    int k_tile_end = k_tile_begin + tiles_per_split;
    if (k_tile_end > k_tiles_total) k_tile_end = k_tiles_total;

    int64_t slab_elems = int64_t(params.M) * int64_t(params.N);
    ElementC* ptr_C_split = params.ptr_C + split_idx * slab_elems;
    ElementSqrSum* ptr_sqsc_split = params.ptr_sqrsum_scratch + split_idx * slab_elems;

    auto layout_A = make_layout(make_shape(params.M, params.K), make_stride(params.stride_A_m, Int<1>{}));
    auto layout_B = make_layout(make_shape(params.N, params.K), make_stride(params.stride_B_k, Int<1>{}));
    auto layout_C = make_layout(make_shape(params.M, params.N), make_stride(params.stride_C_m, Int<1>{}));
    auto layout_sqrsum = make_layout(make_shape(params.M), make_stride(Int<1>{}));

    Tensor A = make_tensor(make_gmem_ptr(params.ptr_A), layout_A);
    Tensor B = make_tensor(make_gmem_ptr(params.ptr_B), layout_B);
    Tensor C = make_tensor(make_gmem_ptr(ptr_C_split), layout_C);
    Tensor SqrSum = make_tensor(make_gmem_ptr(params.ptr_sqrsum), layout_sqrsum);

    auto A_2D = A(append<rank_v<decltype(A)>>(make_coord(_, _), 0));
    auto B_2D = B(append<rank_v<decltype(B)>>(make_coord(_, _), 0));

    CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

    typename CollectiveMainloop::FragGemm tC;
    typename CollectiveMainloop::FragSqrSum tSqrSum;

    mainloop(A_2D, B_2D, tC, tSqrSum, make_coord(blk_m, blk_n), thr_id, k_tile_begin, k_tile_end);

    typename CollectiveMainloop::TiledMMA mma{};
    auto thr_mma = mma.get_slice(thr_id);

    auto cC = make_identity_tensor(C.shape());
    auto gC = local_tile(cC, mma.tile_mnk(), make_coord(blk_m, blk_n, 0), Step<_1, _1, X>{});
    auto copy_c = make_block_2d_copy_D(mma, C);
    auto tCgC = thr_mma.partition_C(gC);

    copy(copy_c, tC, tCgC);

    auto layout_Ssc = make_layout(make_shape(params.M, params.N), make_stride(params.stride_sqsc_m, Int<1>{}));
    Tensor Ssc = make_tensor(make_gmem_ptr(ptr_sqsc_split), layout_Ssc);
    auto cSsc = make_identity_tensor(Ssc.shape());
    auto gSsc = local_tile(cSsc, mma.tile_mnk(), make_coord(blk_m, blk_n, 0), Step<_1, _1, X>{});
    auto copy_sq = make_block_2d_copy_D(mma, Ssc);
    auto tSgSsc = thr_mma.partition_C(gSsc);
    copy(copy_sq, tSqrSum, tSgSsc);
  }
};

}  // namespace cutlass::gemm_sqrsum::kernel
