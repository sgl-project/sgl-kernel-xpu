import hashlib
import os
import statistics

import sgl_kernel  # noqa: F401
import torch

torch.manual_seed(20260929)
device = "xpu"
tokens = 4096
hidden = 6144
start = int(os.getenv("BENCH_VOCAB_START", "0"))
mode = os.getenv("BENCH_MODE", "both")
if mode not in ("both", "baseline"):
    raise ValueError("BENCH_MODE must be both or baseline")
vocab = 50256
weight = torch.randn((vocab, hidden), dtype=torch.bfloat16, device=device)
ids = torch.arange(1024, 1024 + tokens, dtype=torch.int64, device=device)
ids[:4] = torch.tensor(
    [start - 1, start, start + vocab - 1, start + vocab], device=device
)
output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)


def baseline():
    local = (ids - start).clamp(0, vocab - 1)
    result = torch.nn.functional.embedding(local, weight)
    result.masked_fill_(((ids < start) | (ids >= start + vocab)).unsqueeze(-1), 0)
    return result


def candidate():
    torch.ops.sgl_kernel.inkling_vocab_parallel_embedding(
        output, ids, weight, start, start + vocab
    )
    return output


ref = baseline()
if mode == "both":
    got = candidate()
    torch.xpu.synchronize()
    assert torch.equal(ref, got), f"mismatches={(ref != got).sum().item()}"
    digest = hashlib.sha256(got.cpu().view(torch.int16).numpy().tobytes()).hexdigest()
    print(f"shape={tokens}x{hidden} exact=True sha256={digest}")
else:
    print(f"shape={tokens}x{hidden} mode=baseline")

for _ in range(20):
    baseline()
    if mode == "both":
        candidate()
torch.xpu.synchronize()
functions = (("baseline", baseline), ("candidate", candidate))
if mode == "baseline":
    functions = functions[:1]
for label, fn in functions:
    times = []
    for _ in range(int(os.getenv("BENCH_REPEATS", "100"))):
        begin = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    quantiles = statistics.quantiles(times, n=5, method="inclusive")
    print(
        f"{label}_p20_ms={quantiles[0]:.6f} "
        f"{label}_p50_ms={statistics.median(times):.6f} "
        f"{label}_p80_ms={quantiles[3]:.6f}"
    )
