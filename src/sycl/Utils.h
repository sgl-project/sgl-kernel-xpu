#pragma once

#include <ATen/xpu/XPUContext.h>

#include <stdexcept>
#include <type_traits>

#define SYCL_MAX_SUB_GROUP_SIZE dpcppMaxSubGroupSize()

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")
#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

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

static inline int64_t dpcppMaxDSSNum(DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue()) {
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

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                             \
    using scalar_t = type;                      \
    return __VA_ARGS__();                       \
  }

#define SYCL_DISPATCH_FLOATING_TYPES(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                             \
  {                                                                                                         \
    const auto& the_type = TYPE;                                                                            \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                                   \
    switch (_st) {                                                                                          \
      PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                        \
      PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                          \
      PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__) \
      PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__) \
      default:                                                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                     \
    }                                                                                                       \
  }

#define PRIVATE_CASE_TYPE_OUTPLACE(enum_type, type, ...) \
  case enum_type: {                                      \
    using scalar_t = type;                               \
    __VA_ARGS__();                                       \
    break;                                               \
  }

#define SYCL_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  {                                                                                     \
    const auto& the_type = TYPE;                                                        \
    at::ScalarType _st = ::detail::scalar_type(the_type);                               \
    switch (_st) {                                                                      \
      PRIVATE_CASE_TYPE_OUTPLACE(at::ScalarType::Double, double, __VA_ARGS__)           \
      PRIVATE_CASE_TYPE_OUTPLACE(at::ScalarType::Float, float, __VA_ARGS__)             \
      PRIVATE_CASE_TYPE_OUTPLACE(SCALARTYPE1, sycl::ext::oneapi::bfloat16, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_OUTPLACE(SCALARTYPE2, sycl::half, __VA_ARGS__)                  \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                 \
    }                                                                                   \
  }

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}
