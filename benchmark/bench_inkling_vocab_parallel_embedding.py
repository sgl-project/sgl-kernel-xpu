import argparse
import hashlib
import statistics

import sgl_kernel  # noqa: F401
import torch

torch.manual_seed(20260929)
device = "xpu"
parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=4096)
parser.add_argument("--hidden", type=int, default=6144)
parser.add_argument("--vocab", type=int, default=50256)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
parser.add_argument("--repeats", type=int, default=100)
args = parser.parse_args()
if min(args.tokens, args.hidden, args.vocab, args.repeats) <= 0:
    parser.error("tokens, hidden, vocab, and repeats must be positive")
tokens, hidden, vocab, start = args.tokens, args.hidden, args.vocab, args.start
dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
weight = torch.randn((vocab, hidden), dtype=dtype, device=device)
ids = torch.arange(1024, 1024 + tokens, dtype=torch.int64, device=device)
ids[: min(tokens, 4)] = torch.tensor(
    [start, start - 1, start + vocab - 1, start + vocab][: min(tokens, 4)],
    device=device,
)
output = torch.empty((tokens, hidden), dtype=dtype, device=device)


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
got = candidate()
torch.xpu.synchronize()
assert torch.equal(ref, got), f"mismatches={(ref != got).sum().item()}"
digest = hashlib.sha256(got.cpu().view(torch.int16).numpy().tobytes()).hexdigest()
print(
    f"shape={tokens}x{hidden} vocab={vocab} dtype={args.dtype} "
    f"exact=True sha256={digest}"
)

for _ in range(20):
    baseline()
    candidate()
torch.xpu.synchronize()
functions = (("baseline", baseline), ("candidate", candidate))
for label, fn in functions:
    times = []
    for _ in range(args.repeats):
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
