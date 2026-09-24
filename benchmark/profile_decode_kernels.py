"""Minimal single-shape driver for unitrace metric-query profiling of the two
decode conv1d kernels (general-purpose vs. GDN-style no-scan), at bs=128
where the earlier wall-clock benchmark showed GDN-style winning ~1.34x.
Kept to a handful of iterations -- unitrace's metrics CSV grows one row per
kernel invocation, so a full sweep would be excessive for this purpose.
"""

import sys

import sgl_kernel  # noqa: F401
import torch

sys.path.insert(0, "benchmark")
from bench_conv1d_split_compare import (  # noqa: E402
    make_gdn_decode_inputs,
    make_general_decode_inputs,
    run_gdn_decode_fast,
    run_general_decode,
)

BATCH = 128
ITERS = 5

dtype = torch.bfloat16
gen = torch.Generator(device="xpu").manual_seed(42)
gkw = make_general_decode_inputs(BATCH, dtype, gen)
gen2 = torch.Generator(device="xpu").manual_seed(42)
fkw = make_gdn_decode_inputs(BATCH, dtype, gen2)

# warmup (untraced ideally, but unitrace traces everything -- keep low)
for _ in range(2):
    run_general_decode(gkw)
    run_gdn_decode_fast(fkw)
torch.xpu.synchronize()

for _ in range(ITERS):
    run_general_decode(gkw)
torch.xpu.synchronize()

for _ in range(ITERS):
    run_gdn_decode_fast(fkw)
torch.xpu.synchronize()

print("done")
