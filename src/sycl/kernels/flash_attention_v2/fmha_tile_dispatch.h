// Single source of truth for FMHA decode/prefill tile-shape parameters, shared
// by the AOT kernel templates (*.cpp.in, consumed at compile time via
// constexpr lookups keyed on the substituted HEAD_DIM literal) and the
// runtime-JIT wrapper (src/jit/fmha_jit.cpp, consumed at runtime).
//
// Dependency-free (no torch, no SYCL, no cute) so it can be included both by the
// SYCL AOT translation units and by the torch-free `sgl_jit` static library.
// Retuning a tile now touches only this file instead of both the CMake tables
// and the hand-mirrored JIT copy.
#pragma once

namespace sgl {
namespace fmha {

// Round a head dim up to a multiple of 32 (the output-tile head extent unit).
constexpr int round32(int hd) {
  return ((hd + 31) / 32) * 32;
}

// Output-tile head extent: HEAD_DIM rounded to 32, except 512 chunks the head
// into two 256-wide output tiles for occupancy.
constexpr int prefill_tiled_out(int hd) {
  return hd == 512 ? 256 : round32(hd);
}

// KV-tile size for the NON-PAGED decode path (paged decode uses PAGE_SIZE).
// Larger head dims need a smaller KV tile to fit registers/SLM on Level Zero.
constexpr int decode_tiled_kv_np(int hd) {
  return hd >= 256 ? 128 : 512;
}

// Per-HEAD_DIM prefill tile shape (TILED_Q, TILED_KV, NUM_SG). `ok` is false for
// an unsupported head dim so the JIT wrapper can surface a diagnostic; the AOT
// templates only instantiate supported dims so they never observe ok == false.
struct PrefillTile {
  int tiled_q;
  int tiled_kv;
  int num_sg;
  bool ok;
};

// Paged / fp8 prefill (FMHA_PREFILL_TILED_*_<hd> in FMHAPrefillXe20.cmake).
constexpr PrefillTile prefill_paged_tile(int hd) {
  switch (hd) {
    case 64:
      return {128, 64, 8, true};
    case 96:
      return {128, 64, 8, true};
    case 128:
      return {256, 32, 16, true};
    case 192:
      return {256, 64, 32, true};
    case 256:
      return {256, 64, 32, true};
    case 512:
      return {256, 64, 32, true};
    default:
      return {0, 0, 0, false};
  }
}

// Non-paged prefill (FMHA_PREFILL_TILED_*_NP_<hd> in FMHAPrefillXe20.cmake).
constexpr PrefillTile prefill_nopage_tile(int hd) {
  switch (hd) {
    case 64:
      return {128, 64, 8, true};
    case 72:
      return {256, 64, 16, true};
    case 80:
      return {256, 64, 16, true};
    case 96:
      return {256, 64, 16, true};
    case 128:
      return {256, 32, 16, true};
    case 192:
      return {256, 32, 16, true};
    case 256:
      return {128, 64, 16, true};
    case 512:
      return {128, 128, 16, true};
    default:
      return {0, 0, 0, false};
  }
}

}  // namespace fmha
}  // namespace sgl
