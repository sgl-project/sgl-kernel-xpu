import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which(os.environ.get("CXX", "c++")) is None,
    reason="A C++ compiler is required to validate the shared header",
)
def test_aligned_vec_layout_and_access(tmp_path):
    include_dir = Path(__file__).resolve().parents[1] / "include"
    source = tmp_path / "aligned_vec_test.cpp"
    executable = tmp_path / "aligned_vec_test"
    source.write_text(
        """
#include "sgl_kernel/aligned_vec.h"
#include <cstdint>
#include <type_traits>

template <typename T, int N>
class alignas(sizeof(T) * N) LegacyAlignedArray {
  T data[N];
};

template <typename T, int N>
void check() {
  using Vec = aligned_vec<T, N>;
  static_assert(sizeof(Vec) == sizeof(T) * N);
  static_assert(alignof(Vec) == sizeof(T) * N);
  static_assert(sizeof(Vec) == sizeof(LegacyAlignedArray<T, N>));
  static_assert(alignof(Vec) == alignof(LegacyAlignedArray<T, N>));
  static_assert(std::is_standard_layout_v<Vec>);
  static_assert(std::is_trivially_copyable_v<Vec>);
  Vec values{};
  for (int i = 0; i < N; ++i) values[i] = static_cast<T>(i + 1);
  const Vec& copy = values;
  for (int i = 0; i < N; ++i)
    if (copy[i] != static_cast<T>(i + 1) || copy.val[i] != copy[i])
      throw "incorrect vector element";
}

int main() {
  check<std::uint8_t, 1>();
  check<std::uint16_t, 2>();
  check<std::uint16_t, 8>();
  check<float, 4>();
  check<float, 16>();
  check<double, 2>();
}
"""
    )
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-I",
            str(include_dir),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


@pytest.mark.skipif(
    shutil.which("icpx") is None,
    reason="Intel oneAPI icpx is required to compare SYCL vector alignment",
)
def test_aligned_vec_sycl_alignment(tmp_path):
    from torch.utils.cpp_extension import include_paths

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "aligned_vec_sycl_alignment.cpp"
    source.write_text(
        """
#include "MemoryAccess.h"
#include "sgl_kernel/aligned_vec.h"
#include <cstdint>
#include <sycl/sycl.hpp>

template <typename T, typename Storage, int N>
constexpr bool same_alignment = alignof(aligned_vector_t<T, N>) == alignof(sycl::vec<Storage, N>);

template <typename T, typename Storage>
constexpr bool check_widths() {
  static_assert(sizeof(T) == sizeof(Storage));
  return same_alignment<T, Storage, 2> && same_alignment<T, Storage, 4> &&
         same_alignment<T, Storage, 8> && same_alignment<T, Storage, 16>;
}

constexpr bool check_all() {
  return check_widths<std::uint8_t, std::uint8_t>() &&
         check_widths<std::uint16_t, std::uint16_t>() &&
         check_widths<c10::Half, std::uint16_t>() &&
         check_widths<c10::BFloat16, std::uint16_t>() &&
         check_widths<std::int32_t, std::uint32_t>() &&
         check_widths<float, std::uint32_t>() &&
         check_widths<double, std::uint64_t>();
}

static_assert(check_all());
void device_check(sycl::queue& queue) {
  queue.single_task<class AlignmentCheck>([] { static_assert(check_all()); });
}
"""
    )
    includes = [
        root / "include",
        root / "src",
        root / "src" / "sycl",
        *include_paths(),
    ]
    include_flags = [flag for path in includes for flag in ("-I", str(path))]
    subprocess.run(
        ["icpx", "-fsycl", "-std=c++20", "-fsyntax-only", *include_flags, str(source)],
        check=True,
    )
