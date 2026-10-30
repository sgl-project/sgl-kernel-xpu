#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"

#if SYCL_INTEL_TARGET == 20
#include <cute/tensor.hpp>

#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#endif

namespace {

void check_tensor(const at::Tensor& tensor, const at::Device& device, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.device() == device && tensor.is_xpu(), name, " must be on the input XPU device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <int FixedK, int Threads>
struct ConvRotQuantKernel {
  const at::BFloat16* input;
  int8_t* output;
  float* scales;
  int k;
  sycl::local_accessor<float, 1> values;

  void operator()(sycl::nd_item<1> item) const {
    const int k = FixedK ? FixedK : this->k;
    const int tid = item.get_local_id(0);
    const int64_t row = item.get_group(0);
    // Fuse input loading with the first H4 stage, avoiding an SLM round trip.
    for (int b = tid; b < k / 4; b += Threads) {
      const int base = b * 4;
      const float a = static_cast<float>(input[row * k + base]);
      const float b1 = static_cast<float>(input[row * k + base + 1]);
      const float c = static_cast<float>(input[row * k + base + 2]);
      const float d = static_cast<float>(input[row * k + base + 3]);
      values[base] = ((a + b1) + c) - d;
      values[base + 1] = ((a + b1) - c) + d;
      values[base + 2] = ((a - b1) + c) + d;
      values[base + 3] = ((-a + b1) + c) + d;
    }
    sycl::group_barrier(item.get_group());
    // H256 is four regular-H4 stages, not a Sylvester-order FWHT.
    for (int stride = 4; stride < 64; stride *= 4) {
      for (int b = tid; b < k / 4; b += Threads) {
        const int base = (b / stride) * 4 * stride + b % stride;
        const float a = values[base];
        const float b1 = values[base + stride];
        const float c = values[base + 2 * stride];
        const float d = values[base + 3 * stride];
        values[base] = ((a + b1) + c) - d;
        values[base + stride] = ((a + b1) - c) + d;
        values[base + 2 * stride] = ((a - b1) + c) + d;
        values[base + 3 * stride] = ((-a + b1) + c) + d;
      }
      sycl::group_barrier(item.get_group());
    }
    float maximum = 0;
    // Specialized rows retain BF16 results in the producing threads across the reduction.
    constexpr int Chunks = FixedK ? (FixedK + 4 * Threads - 1) / (4 * Threads) : 1;
    at::BFloat16 rounded_values[Chunks][4];
    // Fuse the final H4 stage with BF16 rounding and the local row maximum.
    for (int b = tid; b < k / 4; b += Threads) {
      const int base = (b / 64) * 256 + b % 64;
      const float a = values[base];
      const float b1 = values[base + 64];
      const float c = values[base + 128];
      const float d = values[base + 192];
      const float final_values[4] = {((a + b1) + c) - d, ((a + b1) - c) + d, ((a - b1) + c) + d, ((-a + b1) + c) + d};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float rotated = static_cast<float>(static_cast<at::BFloat16>(final_values[i] * 0.0625f));
        if constexpr (FixedK) {
          rounded_values[(b - tid) / Threads][i] = static_cast<at::BFloat16>(rotated);
        } else {
          values[base + i * 64] = rotated;
        }
        maximum = sycl::fmax(maximum, sycl::fabs(rotated));
      }
    }
    if constexpr (!FixedK) {
      sycl::group_barrier(item.get_group());
    }
    maximum = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>());
    const float scale = sycl::fmax(maximum * (1.0f / 127.0f), 1e-30f);
    if (item.get_local_id(0) == 0) {
      scales[row] = scale;
    }
    const float math_scale = static_cast<float>(static_cast<at::BFloat16>(scale));
    if constexpr (FixedK) {
#pragma unroll
      for (int chunk = 0; chunk < Chunks; ++chunk) {
        const int b = tid + chunk * Threads;
        if (b < k / 4) {
          const int base = (b / 64) * 256 + b % 64;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float quotient = static_cast<float>(
                static_cast<at::BFloat16>(static_cast<float>(rounded_values[chunk][i]) / math_scale));
            output[row * k + base + i * 64] = static_cast<int8_t>(sycl::clamp(sycl::rint(quotient), -128.0f, 127.0f));
          }
        }
      }
    } else {
      for (int j = tid; j < k; j += Threads) {
        const float quotient = static_cast<float>(static_cast<at::BFloat16>(values[j] / math_scale));
        output[row * k + j] = static_cast<int8_t>(sycl::clamp(sycl::rint(quotient), -128.0f, 127.0f));
      }
    }
  }
};

#if SYCL_INTEL_TARGET == 20
using namespace cute;
namespace fusion = cutlass::epilogue::fusion;
using Output = cutlass::bfloat16_t;
using ConvRotTile = Shape<_256, _256, _64>;
using Stride = cute::Stride<int64_t, _1, int64_t>;

struct ConvRotScaleArguments {
  bool has_bias = false;
};

template <class Fragment>
struct ConvRotScale {
  using Arguments = ConvRotScaleArguments;

  CUTLASS_DEVICE Fragment
  operator()(const Fragment& acc, const Fragment& row, const Fragment& col, const Fragment& bias, const Arguments& args)
      const {
    Fragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Fragment::kElements; ++i) {
      const float scale = row[i] * col[i];
      const Output rounded(acc[i] * scale);
      result[i] = args.has_bias ? static_cast<float>(rounded) + bias[i] : static_cast<float>(rounded);
    }
    return result;
  }
};

using ScaleTree = fusion::Sm90EVT<
    fusion::Sm90Compute<ConvRotScale, Output, float, cutlass::FloatRoundStyle::round_to_nearest>,
    fusion::Sm90AccFetch,
    fusion::XeColBroadcast<0, ConvRotTile, float, float, cute::Stride<_1, _0, int64_t>, 1>,
    fusion::XeRowBroadcast<0, ConvRotTile, float, float, cute::Stride<_0, _1, int64_t>, 1>,
    fusion::XeRowBroadcast<0, ConvRotTile, Output, float, cute::Stride<_0, _1, int64_t>, 1>>;
}  // namespace

namespace cutlass::epilogue::fusion {
struct ConvRotOperation : FusionOperation {
  using ElementCompute = float;
};

template <class EpilogueTile>
struct FusionCallbacks<cutlass::epilogue::IntelXeGeneric, ConvRotOperation, ConvRotTile, EpilogueTile> : ScaleTree {
  using ScaleTree::ScaleTree;
};
}  // namespace cutlass::epilogue::fusion

namespace {
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t>>,
    Layout<ConvRotTile>,
    Layout<Shape<_8, _4, _1>, cute::Stride<_4, _1, _0>>>::TiledMMA;
using BaseMainloop = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopXeL1Staged<3>,
    ConvRotTile,
    int8_t,
    Stride,
    int8_t,
    Stride,
    TiledMma,
    void,
    void,
    void,
    cute::identity,
    void,
    void,
    void,
    cute::identity>;
using Callbacks = fusion::FusionCallbacks<
    cutlass::epilogue::IntelXeGeneric,
    fusion::ConvRotOperation,
    ConvRotTile,
    decltype(tile_shape(TiledMma()))>;
using BaseEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::IntelXeGeneric,
    ConvRotTile,
    void,
    int32_t,
    Stride,
    Output,
    Stride,
    Callbacks,
    void,
    void>;
template <class Coord>
CUTLASS_DEVICE auto convrot_tile_coord(Coord coord, int m, int n) {
  const int tiles_m = m / 256 + (m % 256 != 0);
  const int tiles_n = n / 256 + (n % 256 != 0);
  constexpr int GroupM = 4;
  // Keep a small band of M tiles together while traversing N, including a short final band.
  const int64_t linear = int64_t(get<0>(coord)) * tiles_n + get<1>(coord);
  const int64_t band_size = int64_t(GroupM) * tiles_n;
  const int first_m = int(linear / band_size) * GroupM;
  const int height = cute::min(GroupM, tiles_m - first_m);
  const int64_t offset = linear % band_size;
  return make_coord(first_m + int(offset % height), int(offset / height), _, get<3>(coord));
}

struct Mainloop : BaseMainloop {
  template <class D, class A, class B, class C, class Iterator, class Coord>
  CUTLASS_DEVICE void operator()(
      D& accum,
      A,
      B,
      C const& source,
      Iterator iterator,
      int count,
      Coord coord,
      int const& start,
      int thread,
      Params const& params) {
    const int m = size<0>(params.mA_mkl);
    const int n = size<0>(params.mB_nkl);
    const int k = size<1>(params.mA_mkl);
    const int l = size<2>(params.mA_mkl);
    const auto mapped = convrot_tile_coord(coord, m, n);
    const auto a = local_tile(
        make_identity_tensor(make_shape(m, k, l)),
        select<0, 2>(ConvRotTile{}),
        make_coord(get<0>(mapped), _, get<3>(mapped)));
    const auto b = local_tile(
        make_identity_tensor(make_shape(n, k, l)),
        select<1, 2>(ConvRotTile{}),
        make_coord(get<1>(mapped), _, get<3>(mapped)));
    BaseMainloop::operator()(accum, a, b, source, iterator, count, mapped, start, thread, params);
  }
};

struct Epilogue : BaseEpilogue {
  using BaseEpilogue::BaseEpilogue;

  template <class Problem, class Tile, class Coord, class Accumulator, class Mma>
  CUTLASS_DEVICE void operator()(Problem problem, Tile tile, Coord coord, Accumulator accum, Mma mma, int thread) {
    const auto mapped = convrot_tile_coord(coord, get<0>(problem), get<1>(problem));
    BaseEpilogue::operator()(problem, tile, mapped, accum, mma, thread);
  }
};

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, BaseMainloop, BaseEpilogue>;
using GroupedGemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using GroupedGemm = cutlass::gemm::device::GemmUniversalAdapter<GroupedGemmKernel>;
#endif
}  // namespace

SGL_KERNEL_EXPORT std::tuple<at::Tensor, at::Tensor>
convrot_rotate_quantize(const at::Tensor& input, int64_t group_size) {
  check_tensor(input, input.device(), at::kBFloat16, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(group_size == 256, "only ConvRot group_size=256 is supported");
  const int64_t k = input.size(1);
  TORCH_CHECK(k > 0 && k <= 8192 && k % 256 == 0, "K must be a multiple of 256 in [256, 8192]");
  c10::DeviceGuard guard(input.device());
  TORCH_CHECK(is_bmg(), "ConvRot INT8 currently supports BMG (Xe2) devices only");
  auto output = at::empty(input.sizes(), input.options().dtype(at::kChar));
  auto scales = at::empty({input.size(0)}, input.options().dtype(at::kFloat));
  if (input.size(0) > 0) {
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    const auto launch = [&]<int FixedK, int Threads = 256>() {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(k), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(input.size(0) * Threads), sycl::range<1>(Threads)),
            ConvRotQuantKernel<FixedK, Threads>{
                input.data_ptr<at::BFloat16>(),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                static_cast<int>(k),
                values});
      });
    };
    if (k == 1792) {
      launch.template operator()<1792, 256>();
    } else if (k == 3584) {
      launch.template operator()<3584, 256>();
    } else if (k == 5376) {
      launch.template operator()<5376, 512>();
    } else {
      launch.template operator()<0>();
    }
  }
  return {output, scales};
}

SGL_KERNEL_EXPORT at::Tensor convrot_int8_linear_prequant(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& input_scale,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias) {
  check_tensor(input, input.device(), at::kChar, "input");
  check_tensor(weight, input.device(), at::kChar, "weight");
  check_tensor(input_scale, input.device(), at::kFloat, "input_scale");
  check_tensor(weight_scale, input.device(), at::kFloat, "weight_scale");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input and weight must be 2D");
  const int64_t m = input.size(0), k = input.size(1), n = weight.size(0);
  TORCH_CHECK(weight.size(1) == k, "input and weight K must match");
  TORCH_CHECK(k > 0 && k <= 8192 && k % 256 == 0, "K must be a multiple of 256 in [256, 8192]");
  TORCH_CHECK(n > 0 && n % 32 == 0, "N must be a positive multiple of 32");
  TORCH_CHECK(m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max(), "M/N exceed int32 range");
  TORCH_CHECK(input_scale.dim() == 1 && input_scale.numel() == m, "input_scale must be [M]");
  TORCH_CHECK(
      (weight_scale.dim() == 1 || (weight_scale.dim() == 2 && weight_scale.size(1) == 1)) && weight_scale.numel() == n,
      "weight_scale must be [N] or [N, 1]");
  if (bias) {
    check_tensor(*bias, input.device(), at::kBFloat16, "bias");
    TORCH_CHECK(bias->dim() == 1 && bias->numel() == n, "bias must be [N]");
  }
#if SYCL_INTEL_TARGET == 20
  c10::DeviceGuard guard(input.device());
  TORCH_CHECK(is_bmg(), "ConvRot INT8 currently supports BMG (Xe2) devices only");
  auto output = at::empty({m, n}, input.options().dtype(at::kBFloat16));
  if (m == 0) {
    return output;
  }
  Callbacks::Arguments thread{
      {},
      {input_scale.data_ptr<float>(), 0.0f, {_1{}, _0{}, 0}},
      {weight_scale.data_ptr<float>(), 0.0f, {_0{}, _1{}, 0}},
      {bias ? reinterpret_cast<const Output*>(bias->data_ptr()) : nullptr, Output(0.0f), {_0{}, _1{}, 0}},
      {bias.has_value()}};
  const auto launch = [&]<class DeviceGemm>() {
    typename DeviceGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
        {input.data_ptr<int8_t>(), {k, _1{}, 0}, weight.data_ptr<int8_t>(), {k, _1{}, 0}},
        {thread, nullptr, {n, _1{}, 0}, reinterpret_cast<Output*>(output.data_ptr()), {n, _1{}, 0}}};
    DeviceGemm gemm;
    auto status = gemm.can_implement(args);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "ConvRot INT8 GEMM cannot implement shape: ",
        cutlassGetStatusString(status));
    auto workspace =
        at::empty({static_cast<int64_t>(DeviceGemm::get_workspace_size(args))}, input.options().dtype(at::kByte));
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    status = gemm.run(args, workspace.data_ptr(), &queue);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess, "ConvRot INT8 GEMM launch failed: ", cutlassGetStatusString(status));
  };
  if (n == 7168 && k == 5376 && m >= 16384) {
    launch.template operator()<GroupedGemm>();
  } else {
    launch.template operator()<Gemm>();
  }
  return output;
#else
  TORCH_CHECK(false, "ConvRot INT8 GEMM currently requires a BMG (Xe2) build");
#endif
}
