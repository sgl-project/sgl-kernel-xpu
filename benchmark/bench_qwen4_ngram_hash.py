import argparse

import sgl_kernel
import torch
import triton

# Qwen3.8-Flash-Next PLE n-gram ids: 3-token windows, 16 heads, used in decode and NEXTN verify.
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NGRAM_HEADS = 16
EOS = 151645
UNIGRAM_VOCAB = 160000
TOKENS = [1, 4, 16, 64, 128, 256, 1024, 8192, 65536, 524288]
PROVIDERS = ["sgl_kernel", "torch"]

times_us = {}


def shift_right_ignore_eos(tensor, n, eos):
    # Copied from Qwen4ExpNGramEmbedding._shift_right_ignore_eos.
    batch_size, seq_len = tensor.shape
    idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
    eos_pos = torch.where(tensor == eos, idx, -1)
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat(
        [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1
    )
    pos_in_segment = idx.unsqueeze(0) - (prev_eos + 1)
    src_idx = idx - n
    gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
    shifted = tensor.gather(dim=1, index=gather_idx)
    valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid_mask, shifted, tensor.new_full((), eos))


def ngram_hash_torch(contexts, multipliers, vocab_sizes, offsets):
    shifted = [contexts] + [
        shift_right_ignore_eos(contexts, s, EOS) for s in range(1, NGRAM_SIZE)
    ]
    blocks = []
    for ngram in range(2, NGRAM_SIZE + 1):
        start = (ngram - 2) * HEADS_PER_NGRAM
        end = start + HEADS_PER_NGRAM
        mix = shifted[0] * multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * multipliers[pos])
        ids = torch.remainder(
            mix[:, -1:].unsqueeze(-1), vocab_sizes[start:end].view(1, 1, -1)
        )
        blocks.append((ids + offsets[start:end].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)


def bytes_moved(tokens):
    # Reads the context window and the tables, writes the ids; all int64.
    return (tokens * NGRAM_SIZE + 3 + 2 * NGRAM_HEADS + tokens * NGRAM_HEADS) * 8


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=TOKENS,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="qwen4-ngram-hash-performance",
        args={},
    )
)
def benchmark(tokens, provider):
    device = "xpu"
    torch.manual_seed(0)
    contexts = torch.randint(0, UNIGRAM_VOCAB, (tokens, NGRAM_SIZE), device=device)
    contexts = torch.where(
        torch.rand(contexts.shape, device=device) < 0.1, EOS, contexts
    )
    multipliers = (
        torch.randint(1, 2**62 // UNIGRAM_VOCAB, (NGRAM_SIZE,), device=device) | 1
    )
    vocab_sizes = torch.randint(100_000, 5_000_000, (NGRAM_HEADS,), device=device)
    offsets = torch.cumsum(vocab_sizes, 0) - vocab_sizes
    out = torch.empty(tokens, NGRAM_HEADS, dtype=torch.long, device=device)

    if provider == "sgl_kernel":

        def fn():
            sgl_kernel.qwen4_ngram_hash(
                contexts, multipliers, vocab_sizes, offsets, EOS, out=out
            )

    else:

        def fn():
            ngram_hash_torch(contexts, multipliers, vocab_sizes, offsets)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    times_us[provider] = ms * 1000
    if provider == PROVIDERS[-1]:
        t = times_us
        gbps = bytes_moved(tokens) / (t["sgl_kernel"] * 1e-6) / 1e9
        print(
            f"{tokens:>8}{t['sgl_kernel']:>14.2f}{t['torch']:>14.2f}"
            f"{gbps:>14.2f}{t['torch'] / t['sgl_kernel']:>14.2f}",
            flush=True,
        )
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    args = parser.parse_args()
    benchmark.benchmarks.x_vals = args.tokens

    print("qwen4 PLE n-gram hash: median time (triton do_bench), us\n")
    print(
        f"{'tokens':>8}{'sgl_kernel':>14}{'torch':>14}{'sgl_GB/s':>14}{'vs_torch':>14}",
        flush=True,
    )
    benchmark.run(print_data=False)
