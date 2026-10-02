import argparse
import json
from dataclasses import dataclass

import torch
import triton
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@dataclass
class Case:
    name: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    rel_bias: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    out: torch.Tensor
    window_size: tuple[int, int]

    def run(self):
        return flash_attn_with_kvcache(
            self.q,
            self.k,
            self.v,
            cache_seqlens=self.cache_seqlens,
            page_table=self.page_table,
            cu_seqlens_q=self.cu_seqlens_q,
            max_seqlen_q=self.q.shape[0],
            max_seqlen_k=self.q.shape[0],
            softmax_scale=1.0 / self.q.shape[-1],
            causal=True,
            window_size=self.window_size,
            return_softmax_lse=False,
            out=self.out,
            rel_bias=self.rel_bias,
        )


def make_case(
    name: str,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    extent: int,
    window_size: tuple[int, int],
    dtype: torch.dtype,
) -> Case:
    page_size = 128
    pages = (seq_len + page_size - 1) // page_size
    device = torch.device("xpu")
    q = torch.randn(seq_len, q_heads, 128, dtype=dtype, device=device)
    k = torch.randn(pages, page_size, kv_heads, 128, dtype=dtype, device=device)
    v = torch.randn_like(k)
    rel_bias = torch.randn(seq_len, q_heads, extent, dtype=dtype, device=device)
    return Case(
        name=name,
        q=q,
        k=k,
        v=v,
        rel_bias=rel_bias,
        page_table=torch.arange(pages, dtype=torch.int32, device=device).view(1, -1),
        cache_seqlens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        out=torch.empty_like(q),
        window_size=window_size,
    )


def measure(fn, warmup: int, rep: int) -> dict[str, float]:
    median, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    return {"median_ms": median, "p20_ms": p20, "p80_ms": p80}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--tails", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20260923)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    local = make_case("local_swa", 4096, 16, 4, 512, (511, 0), dtype)
    global_gqa = make_case("global_gqa", 4096, 16, 2, 1024, (-1, -1), dtype)

    results = {
        "dtype": args.dtype,
        "device": torch.xpu.get_device_name(),
        "local_swa": measure(local.run, args.warmup, args.rep),
        "global_gqa": measure(global_gqa.run, args.warmup, args.rep),
    }

    def reduced_six_layers():
        for _ in range(5):
            local.run()
        global_gqa.run()

    results["reduced_six_layers"] = measure(reduced_six_layers, args.warmup, args.rep)
    measured = results["reduced_six_layers"]["median_ms"]
    results["reduced_six_layers"]["lmproj_ms"] = 4.517
    results["reduced_six_layers"]["lmproj_efficiency_pct"] = 451.7 / measured
    results["reduced_six_layers"]["target_ms"] = 7.53
    results["reduced_six_layers"]["target_met"] = measured <= 7.53

    if args.tails:
        tail_local = make_case("tail_local_swa", 4093, 16, 4, 513, (511, 0), dtype)
        tail_global = make_case("tail_global_gqa", 4091, 16, 2, 1025, (-1, -1), dtype)
        results["tail_local_swa"] = measure(tail_local.run, args.warmup, args.rep)
        results["tail_global_gqa"] = measure(tail_global.run, args.warmup, args.rep)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
