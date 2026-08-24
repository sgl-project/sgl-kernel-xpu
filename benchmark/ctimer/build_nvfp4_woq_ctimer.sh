#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-/home/sdp/miniforge3/envs/sglang/bin/python}"
CXX="${CXX:-icpx}"
CUTLASS="$ROOT/build/_deps/repo-cutlass-sycl-src"
TORCH_ROOT="$($PYTHON -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')"
TORCH_LIB="$TORCH_ROOT/lib"
PYTHON_INCLUDE="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
ABI="$($PYTHON -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
OUT="$HERE/nvfp4_ctimer_ext$EXT_SUFFIX"

"$CXX" \
  -fsycl -shared -fPIC -O2 -std=c++17 \
  -DTORCH_EXTENSION_NAME=nvfp4_ctimer_ext \
  -DCUTLASS_ENABLE_SYCL \
  -DCUTLASS_SYCL_PROFILING_ENABLED \
  -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  -I"$PYTHON_INCLUDE" \
  -I"$TORCH_ROOT/include" \
  -I"$TORCH_ROOT/include/torch/csrc/api/include" \
  -I"$CUTLASS/include" \
  -I"$CUTLASS/tools/util/include" \
  "$HERE/bench_nvfp4_woq_ctimer.cpp" \
  "$ROOT/build/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so" \
  "$ROOT/build/src/libsgl-ops-sycl-blockwise_moe_mxfp8.so" \
  -L"$TORCH_LIB" \
  -ltorch_python -ltorch_xpu -ltorch_cpu -ltorch -lc10_xpu -lc10 \
  -Wl,-rpath,"$TORCH_LIB" \
  -Wl,-rpath,"$ROOT/build/src" \
  -o "$OUT"

echo "built $OUT"
