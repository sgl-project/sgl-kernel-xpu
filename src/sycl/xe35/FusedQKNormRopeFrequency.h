#pragma once

#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

namespace at::native::xpu {

template <int HeadDim>
inline std::array<float, HeadDim / 2>
qk_rope_frequencies_xe35(float base, int rotary_dim, float factor, float low, float high) {
  std::array<float, HeadDim / 2> frequencies{};
  for (int half_dim = 0; half_dim < rotary_dim / 2; ++half_dim) {
    float frequency = static_cast<float>(std::pow(static_cast<double>(base), -2.0 * half_dim / rotary_dim));
    if (factor != 1.0f) {
      const float high_adjusted = std::fabs(low - high) <= 1e-6f ? high + 0.001f : high;
      const float ramp = std::clamp((2.0f * half_dim - low) / (high_adjusted - low), 0.0f, 1.0f);
      frequency = (frequency / factor) * (1.0f - ramp) + frequency * ramp;
    }
    frequencies[half_dim] = frequency;
  }
  return frequencies;
}

struct QKRopeFrequencyTable_xe35 {
  sycl::context context;
  sycl::device device;
  std::array<float, 4> parameters;
  int rotary_dim;
  float* data;

  QKRopeFrequencyTable_xe35(const sycl::queue& queue, std::array<float, 4> parameters, int rotary_dim)
      : context(queue.get_context()),
        device(queue.get_device()),
        parameters(parameters),
        rotary_dim(rotary_dim),
        data(sycl::malloc_shared<float>(rotary_dim / 2, device, context)) {
    TORCH_CHECK(data != nullptr, "Failed to allocate QK RoPE frequency table");
    const auto values =
        qk_rope_frequencies_xe35<256>(parameters[0], rotary_dim, parameters[1], parameters[2], parameters[3]);
    std::copy_n(values.data(), rotary_dim / 2, data);
  }

  ~QKRopeFrequencyTable_xe35() {
    sycl::free(data, context);
  }

  bool matches(const sycl::queue& queue, const std::array<float, 4>& requested, int dim) const {
    return rotary_dim == dim && parameters == requested && context == queue.get_context() &&
           device == queue.get_device();
  }
};

inline const float*
qk_rope_cached_frequencies_xe35(const sycl::queue& queue, float base, int rotary_dim, float factor, float low, float high) {
  // A norm-only call (rotary_dim == 0) rotates nothing and has no frequencies
  // to cache; the kernels never read them, so take the inline-storage path.
  if (rotary_dim < 2) return nullptr;
  if (!queue.get_device().has(sycl::aspect::usm_shared_allocations)) return nullptr;
  const std::array<float, 4> parameters{base, factor, low, high};
  static std::mutex mutex;
  static std::vector<std::shared_ptr<QKRopeFrequencyTable_xe35>> tables;
  thread_local std::shared_ptr<QKRopeFrequencyTable_xe35> last;
  if (last && last->matches(queue, parameters, rotary_dim)) return last->data;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& table : tables) {
    if (table->matches(queue, parameters, rotary_dim)) {
      last = table;
      return table->data;
    }
  }
  auto table = std::make_shared<QKRopeFrequencyTable_xe35>(queue, parameters, rotary_dim);
  tables.push_back(table);
  last = table;
  return table->data;
}

template <int HeadDim, bool UseTable>
using QKRopeFrequencyStorage_xe35 = std::conditional_t<UseTable, const float*, std::array<float, HeadDim / 2>>;

template <int HeadDim, bool UseTable>
inline QKRopeFrequencyStorage_xe35<HeadDim, UseTable>
qk_rope_frequency_storage_xe35(const float* table, float base, int rotary_dim, float factor, float low, float high) {
  if constexpr (UseTable)
    return table;
  else
    return qk_rope_frequencies_xe35<HeadDim>(base, rotary_dim, factor, low, high);
}

}  // namespace at::native::xpu
