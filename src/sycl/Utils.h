#pragma once

#include <ATen/Dispatch_v2.h>
#include <ATen/xpu/XPUContext.h>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "jit/jit_arch.h"  // sgl::jit::Arch, consumed by arch_profile()

#define SYCL_MAX_SUB_GROUP_SIZE dpcppMaxSubGroupSize()

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")
#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == c10::ArrayRef<int64_t>({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")
#define CHECK_INPUT(x) \
  CHECK_DEVICE(x);     \
  CHECK_CONTIGUOUS(x);
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_DEVICE(x);                         \
  CHECK_LAST_DIM_CONTIGUOUS(x)

// Use AT_DISPATCH_* for ATen/c10 types; SYCL_DISPATCH_* only when kernel
// operations require SYCL types (e.g. sycl::vec), not merely because it runs on XPU.
// Project helper: FP32/FP16/BF16, no Double, no SYCL type mapping.
#define AT_DISPATCH_FLOATING_TYPES_V2(TYPE, NAME, ...) \
  AT_DISPATCH_V2(TYPE, NAME, AT_WRAP(__VA_ARGS__), at::kFloat, at::kHalf, at::kBFloat16)

// INPUT_T is an ATen/c10 type; weight_t is INPUT_T or float, not an independent dtype dispatch.
// Other weight dtypes throw. TYPE is evaluated once; callback returns stay inside this expression.
#define AT_DISPATCH_WEIGHT_TYPES(INPUT_T, TYPE, NAME, ...)                                                            \
  [&]() -> decltype(auto) {                                                                                           \
    const auto _weight_type = TYPE;                                                                                   \
    if (_weight_type == c10::CppTypeToScalarType<INPUT_T>::value) {                                                   \
      using weight_t = INPUT_T;                                                                                       \
      return (__VA_ARGS__)();                                                                                         \
    }                                                                                                                 \
    TORCH_CHECK(                                                                                                      \
        _weight_type == at::kFloat, NAME, ": weight dtype must match input dtype or be float32, got ", _weight_type); \
    using weight_t = float;                                                                                           \
    return (__VA_ARGS__)();                                                                                           \
  }()

template <typename T>
struct sycl_type {
  using type = T;
};

template <>
struct sycl_type<at::Half> {
  using type = sycl::half;
};

template <>
struct sycl_type<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

template <typename T>
using sycl_type_t = typename sycl_type<T>::type;

#define SYCL_DISPATCH_BODY(...)                  \
  AT_WRAP([&] {                                  \
    using sycl_scalar_t = sycl_type_t<scalar_t>; \
    return [&] {                                 \
      using scalar_t = sycl_scalar_t;            \
      return (__VA_ARGS__)();                    \
    }();                                         \
  })

// FP16/BF16.
#define SYCL_DISPATCH_FLOATING_TYPES_V1(TYPE, NAME, ...) \
  AT_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, SYCL_DISPATCH_BODY(__VA_ARGS__))

// FP32/FP16/BF16.
#define SYCL_DISPATCH_FLOATING_TYPES_V2(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_V2(TYPE, NAME, SYCL_DISPATCH_BODY(__VA_ARGS__))

// FP64/FP32/FP16/BF16.
#define SYCL_DISPATCH_FLOATING_TYPES_V3(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, TYPE, NAME, SYCL_DISPATCH_BODY(__VA_ARGS__))

#define SGL_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)        \
  [&]() -> decltype(auto) {                              \
    if (BOOL_V) {                                        \
      [[maybe_unused]] constexpr bool BOOL_NAME = true;  \
      return (__VA_ARGS__)();                            \
    } else {                                             \
      [[maybe_unused]] constexpr bool BOOL_NAME = false; \
      return (__VA_ARGS__)();                            \
    }                                                    \
  }()

#define SGL_DISPATCH_BOOL2(BOOL_V1, BOOL_NAME1, BOOL_V2, BOOL_NAME2, ...) \
  SGL_DISPATCH_BOOL(                                                      \
      BOOL_V1, BOOL_NAME1, [&]() -> decltype(auto) { return SGL_DISPATCH_BOOL(BOOL_V2, BOOL_NAME2, __VA_ARGS__); })

#define SGL_DISPATCH_BOOL3(BOOL_V1, BOOL_NAME1, BOOL_V2, BOOL_NAME2, BOOL_V3, BOOL_NAME3, ...) \
  SGL_DISPATCH_BOOL(BOOL_V1, BOOL_NAME1, [&]() -> decltype(auto) {                             \
    return SGL_DISPATCH_BOOL2(BOOL_V2, BOOL_NAME2, BOOL_V3, BOOL_NAME3, __VA_ARGS__);          \
  })

// Expand only the supplied widths (one to five), e.g. (1, 2, 4, 8).
#define SGL_DISPATCH_VEC_SIZE_CASE(SIZE, ...) \
  case SIZE: {                                \
    constexpr uint32_t VEC_SIZE = SIZE;       \
    return (__VA_ARGS__)();                   \
  }

#define SGL_VEC_SIZE_CASES_1(BODY, A) SGL_DISPATCH_VEC_SIZE_CASE(A, BODY)
#define SGL_VEC_SIZE_CASES_2(BODY, A, ...) SGL_DISPATCH_VEC_SIZE_CASE(A, BODY) SGL_VEC_SIZE_CASES_1(BODY, __VA_ARGS__)
#define SGL_VEC_SIZE_CASES_3(BODY, A, ...) SGL_DISPATCH_VEC_SIZE_CASE(A, BODY) SGL_VEC_SIZE_CASES_2(BODY, __VA_ARGS__)
#define SGL_VEC_SIZE_CASES_4(BODY, A, ...) SGL_DISPATCH_VEC_SIZE_CASE(A, BODY) SGL_VEC_SIZE_CASES_3(BODY, __VA_ARGS__)
#define SGL_VEC_SIZE_CASES_5(BODY, A, ...) SGL_DISPATCH_VEC_SIZE_CASE(A, BODY) SGL_VEC_SIZE_CASES_4(BODY, __VA_ARGS__)
#define SGL_VEC_SIZE_SELECT(_1, _2, _3, _4, _5, CASES, ...) CASES
#define SGL_VEC_SIZE_UNPACK(...) __VA_ARGS__
#define SGL_VEC_SIZE_CASES(BODY, ...) \
  SGL_VEC_SIZE_SELECT(                \
      __VA_ARGS__,                    \
      SGL_VEC_SIZE_CASES_5,           \
      SGL_VEC_SIZE_CASES_4,           \
      SGL_VEC_SIZE_CASES_3,           \
      SGL_VEC_SIZE_CASES_2,           \
      SGL_VEC_SIZE_CASES_1)           \
  (BODY, __VA_ARGS__)

#define SGL_DISPATCH_VEC_SIZE_SET(SIZE, SIZES, ...)                 \
  [&] {                                                             \
    const auto _vec_size = SIZE;                                    \
    switch (_vec_size) {                                            \
      SGL_VEC_SIZE_CASES((__VA_ARGS__), SGL_VEC_SIZE_UNPACK SIZES)  \
      default:                                                      \
        TORCH_CHECK(false, "Unsupported vector size: ", _vec_size); \
    }                                                               \
  }()

// Default widths: 1/2/4/8/16. Other widths throw.
#define SGL_DISPATCH_VEC_SIZE(SIZE, ...) SGL_DISPATCH_VEC_SIZE_SET(SIZE, (1, 2, 4, 8, 16), __VA_ARGS__)

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

// BF16 MoE grouped-GEMM tile-selection cutoff. Python mirrors this threshold
// when choosing the BF16 fused-activation path; update both when retuning.
constexpr int64_t MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD = int64_t(4096) * 4096;

static inline void barrier() {
  // Compiler + hardware memory fence for work-group–level synchronization.
  asm volatile("lsc_fence.ugm.none.group\n" ::: "memory");
  asm volatile("barrier\n" ::: "memory");
}

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture get_device_architecture(at::DeviceIndex device_index = -1) {
  auto device_id = (device_index == -1) ? c10::xpu::current_device() : device_index;
  auto raw_device = c10::xpu::get_raw_device(device_id);
  return raw_device.get_info<syclex::info::device::architecture>();
}

static inline bool is_bmg(at::DeviceIndex device_index = -1) {
  return get_device_architecture(device_index) == syclex::architecture::intel_gpu_bmg_g21 ||
         get_device_architecture(device_index) == syclex::architecture::intel_gpu_bmg_g31;
}

// Runtime-JIT architecture code (== static_cast<int>(sgl::jit::Arch); the value
// selects the arch_profile used when compiling a kernel). Classifies the real
// SYCL device architecture, mirroring Device.cpp's query_device switch:
// bmg_g21/g31 => BMG/Xe20. Returned as int so callers pass it straight into the
// launch entry points.
static inline int jit_arch_code(at::DeviceIndex device_index = -1) {
  switch (get_device_architecture(device_index)) {
    case syclex::architecture::intel_gpu_bmg_g21:
    case syclex::architecture::intel_gpu_bmg_g31:
      return static_cast<int>(sgl::jit::Arch::BMG);
    // case syclex::architecture::intel_gpu_<xe35>:
    //   return static_cast<int>(sgl::jit::Arch::XE3P);  // Xe35, when the enum lands
    default:
      // Unknown architecture: fail loudly instead of silently compiling BMG
      // device code, which would produce wrong results at runtime.
      TORCH_CHECK(
          false,
          "Runtime-JIT does not support this XPU architecture yet. "
          "Set SGLANG_DISABLE_SYCL_JIT=1 or rebuild with USE_SYCL_JIT=OFF to "
          "use the AOT kernels.");
      return static_cast<int>(sgl::jit::Arch::BMG);  // unreachable
  }
}

using DeviceId = at::DeviceIndex;

static inline DeviceId dpcppGetDeviceIdOfCurrentQueue() {
  return at::xpu::current_device();
}

static inline sycl::queue& dpcppGetCurrentQueue() {
  return at::xpu::getCurrentXPUStream().queue();
}

template <class KernelClass>
static int64_t dpcppMaxWorkGroupSize(at::DeviceIndex dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto q = c10::xpu::getCurrentXPUStream(dev_id).queue();
  auto ctx = q.get_context();
  auto dev = q.get_device();

  auto kid = ::sycl::get_kernel_id<KernelClass>();
  auto kbundle = ::sycl::get_kernel_bundle<::sycl::bundle_state::executable>(ctx, {kid});

  ::sycl::kernel k = kbundle.get_kernel(kid);
  return k.get_info<::sycl::info::kernel_device_specific::work_group_size>(dev);
}

template <class KernelClass>
static int64_t dpcppMaxWorkGroupSize(KernelClass /*kfn*/, at::DeviceIndex dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  return dpcppMaxWorkGroupSize<KernelClass>(dev_id);
}

static inline int64_t dpcppMaxWorkGroupSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t dpcppMaxSubGroupSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto subgroup_sizes = dev_prop->sub_group_sizes;
  int64_t max_val = 0;
  for (auto i : subgroup_sizes) {
    if (i > max_val) max_val = i;
  }
  return max_val;
}

static inline int64_t dpcppMinSubGroupSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto subgroup_sizes = dev_prop->sub_group_sizes;
  int64_t min_val = dev_prop->max_work_group_size;
  for (auto i : subgroup_sizes) {
    if (i < min_val) min_val = i;
  }
  return min_val;
}

static inline int64_t dpcppMaxComputeUnitSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_compute_units;
}

static inline int64_t dpcppGpuEuCount(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count;
}

static inline int64_t dpcppGpuEuCountPerSubslice(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count_per_subslice;
}

static inline int64_t dpcppGpuSubsliceCount(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  int64_t eu_count_per_subslice = dpcppGpuEuCountPerSubslice(dev_id);
  if (eu_count_per_subslice <= 0) return 1;
  return std::max<int64_t>(1, dpcppGpuEuCount(dev_id) / eu_count_per_subslice);
}

static inline int64_t dpcppGpuEuSimdWidth(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_simd_width;
}

static inline int64_t dpcppGpuHWThreadsPerEU(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_hw_threads_per_eu;
}

static inline bool dpcppSupportAtomic64(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->has_atomic64;
}

static inline int64_t dpcppMaxWorkItemsPerTile(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t eu_cnt = dev_prop->gpu_eu_count;
  int64_t simd_width = dpcppMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return eu_cnt * simd_width * hw_threads;
}

static inline int64_t dpcppMaxWorkItemsPerEU(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = dpcppMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return simd_width * hw_threads;
}

static inline int64_t dpcppMaxDSSNum([[maybe_unused]] DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  // TODO: We need to got this info from DPC++ Runtime
  // Hardcode to 32 for ATS
  int64_t dss_num = 32;
  return dss_num;
}

static inline size_t dpcppGlobalMemSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->global_mem_size;
}

static inline int64_t dpcppLocalMemSize(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->local_mem_size;
}

template <typename T>
uint32_t dpcppPrefVectorWidth(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->preferred_vector_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->preferred_vector_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->preferred_vector_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->preferred_vector_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->preferred_vector_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->preferred_vector_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->preferred_vector_width_half;
  }
  throw std::invalid_argument("Invalid data type to fetch preferred vector width!");
}

template <typename T>
uint32_t dpcppNativeVectorWidth(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->native_vector_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->native_vector_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->native_vector_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->native_vector_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->native_vector_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->native_vector_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->native_vector_width_half;
  }
  throw std::invalid_argument("Invalid data type to fetch native vector width!");
}

template <typename Func, typename... Args>
int get_min(Func limit_func, int X, Args*... args) {
  X = std::min({X, limit_func(X, args)...});
  return X;
}

inline void check_shape(const at::Tensor& a, const at::Tensor& b, const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ", a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name, ".size(", i, ")");
  }
}

#define CHECK_SAME_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

// Reciprocal with zero check. If x is zero, return zero; else return 1/x.
template <typename T>
inline T safe_recip(T x) {
  const auto nz = (x != T(0));
  const T safe_d = sycl::select(T(1), x, nz);
  const T inv = sycl::native::recip(safe_d);
  return sycl::select(T(0), inv, nz);
}

inline int nextPowerOf2(uint32_t a) {
  if (a <= 1) return 1;
  return 1 << (32 - __builtin_clz(a - 1));
};
inline int round_up_headdim(int head_size) {
  if (head_size <= 64) return 64;
  if (head_size <= 96) return 96;
  if (head_size <= 128) return 128;
  if (head_size <= 192) return 192;
  if (head_size <= 256) return 256;
  if (head_size <= 512) return 512;
  return 512;
};

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};

template <>
struct DtypeInfo<c10::Float8_e4m3fn> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vec {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) {
    return val[index];
  }

  scalar_t const& operator[](int index) const {
    return val[index];
  }
};
