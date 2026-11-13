#!/usr/bin/env python3
"""One entrypoint for the hal Job. MODE selects smoke, numeric, generate, bench, or benchy."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MARKER = "SGL_KERNEL_XPU_GPTQ_ENTRY"
PROMPT = "Write a one-sentence hardware note."
_max_new = os.environ.get("MAX_NEW_TOKENS", "").strip()
MAX_NEW_TOKENS = int(_max_new) if _max_new else 32
TEMPERATURE = 0
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAU-GPTQ-Int4-g128-tuned",
)
RECORD_PREFIX = "SGL_GPTQ_GENERATION_RECORD"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


# 64 and 128 are the paged XPU kernel sizes. 1 stays available for the
# token-addressed no_buffer cache. Context matches the live server.
PAGE_SIZE = _int_env("SGL_GPTQ_PAGE_SIZE", 64)
CONTEXT_LENGTH = _int_env("SGL_GPTQ_CONTEXT_LENGTH", 131072)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def require_xpu():
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        fail("XPU is required")
    torch.xpu.set_device(0)
    return torch


def require_greedy() -> None:
    raw = os.environ.get("TEMPERATURE", str(TEMPERATURE))
    try:
        value = float(raw)
    except ValueError:
        fail(f"temperature must be 0, got {raw}")
    if value != 0.0:
        fail(f"temperature must be 0, got {value}")
    sampling = os.environ.get("SAMPLING", "greedy")
    if sampling not in ("greedy", "0", ""):
        fail(f"sampling must be greedy, got {sampling}")
    prompt = os.environ.get("SGL_GPTQ_PROMPT", PROMPT)
    if prompt == "":
        fail("prompt must be non-empty")


def pack_rows(q_uint8):
    import torch

    k, n = q_uint8.shape
    out = torch.zeros(k // 8, n, dtype=torch.int32)
    for i in range(8):
        out |= q_uint8[i::8].to(torch.int32) << (i * 4)
    return out


def pack_cols(z_uint8):
    import torch

    g, n = z_uint8.shape
    out = torch.zeros(g, n // 8, dtype=torch.int32)
    for i in range(8):
        out |= z_uint8[:, i::8].to(torch.int32) << (i * 4)
    return out


def make_case(m, n, k, dtype, device, g_idx, zero_format, seed, round_weights=False):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    weight = torch.randn(k, n, generator=gen)
    act = torch.randn(m, k, generator=gen)
    groups = int(g_idx.max().item()) + 1
    scales = torch.empty(groups, n)
    stored = torch.empty(groups, n, dtype=torch.int32)
    q = torch.empty(k, n)
    for gi in range(groups):
        rows = weight[g_idx == gi]
        wmin = rows.min(dim=0).values
        wmax = rows.max(dim=0).values
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-4)
        eff = (-wmin / scale).round()
        if zero_format == 1:
            eff = eff.clamp(1, 15)
            stored[gi] = (eff - 1).to(torch.int32)
        else:
            eff = eff.clamp(0, 15)
            stored[gi] = eff.to(torch.int32)
        scales[gi] = scale
        sel = g_idx == gi
        q[sel] = (weight[sel] / scale + eff).round().clamp(0, 15)
    eff_rows = stored[g_idx].to(torch.float32)
    if zero_format == 1:
        eff_rows = eff_rows + 1
    act_q = act.to(dtype)
    scales_q = scales.to(dtype)
    # XMX multiplies bfloat16 weights. Round the dequantized matrix to that
    # dtype before the reference product so the comparison is the tensor-core
    # result, not an fp32-weight product.
    weight_f = (q - eff_rows) * scales_q.float()[g_idx]
    if round_weights:
        weight_f = weight_f.to(dtype).float()
    ref = (act_q.float() @ weight_f).to(dtype)
    return (
        act_q.to(device),
        pack_rows(q.to(torch.uint8)).to(device),
        pack_cols(stored.to(torch.uint8)).to(device),
        scales_q.to(device),
        g_idx.to(device=device, dtype=torch.int32),
        ref,
    )


def within_tolerance(out, ref) -> bool:
    import torch

    got = out.to(ref.dtype).float().cpu()
    expect = ref.float().cpu()
    diff = (got - expect).abs()
    limit = torch.maximum(0.01 * expect.abs(), torch.full_like(expect, 0.02))
    return bool(torch.all(diff <= limit).item())


def smoke() -> None:
    require_xpu()
    from sgl_kernel import gptq_entry_probe

    gptq_entry_probe()


def numeric() -> None:
    import torch

    require_xpu()
    from sgl_kernel import gptq_gemm

    device = torch.device("xpu")
    # Transpose is cached by storage address. The caching allocator recycles
    # those addresses, so every probe tensor has to stay alive until the end.
    hold = []
    k, n = 256, 32
    groups = k // 128
    contiguous = torch.arange(k) // 128
    reversed_idx = (groups - 1) - contiguous
    cases = [
        (1, torch.float16, contiguous, 1, "contiguous"),
        (8, torch.float16, contiguous, 1, "contiguous"),
        (1, torch.bfloat16, contiguous, 1, "contiguous"),
        (4, torch.bfloat16, contiguous, 1, "contiguous"),
        (1, torch.float16, reversed_idx, 1, "noncontiguous"),
        (4, torch.bfloat16, reversed_idx, 2, "noncontiguous"),
        (1, torch.float16, contiguous, 2, "contiguous"),
        (4, torch.bfloat16, contiguous, 2, "contiguous"),
    ]
    passed = 0
    failed = 0
    for idx, (m, dtype, g_idx, zero_format, index_name) in enumerate(cases):
        label = f"{dtype} M={m} g_idx={index_name} zero_format={zero_format}"
        act, qweight, qzeros, scales, g, ref = make_case(
            m,
            n,
            k,
            dtype,
            device,
            g_idx,
            zero_format,
            seed=1000 + idx,
            round_weights=False,
        )
        out = gptq_gemm(act, qweight, qzeros, scales, g, False, 4, zero_format)
        torch.xpu.synchronize()
        hold.append((act, qweight, qzeros, scales, g, out))
        if within_tolerance(out, ref):
            passed += 1
            print(f"numeric {label} pass")
        else:
            failed += 1
            got = out.float().cpu()
            expect = ref.float().cpu()
            diff = (got - expect).abs()
            print(
                f"numeric {label} fail max={float(diff.max()):.6g} "
                f"mean={float(diff.mean()):.6g} "
                f"got={got.flatten()[:4].tolist()} ref={expect.flatten()[:4].tolist()}"
            )

    # K=2048 hits only the 4-wide pipeline. K=2176 also hits its tail.
    for wide_k, wide_m in ((2048, 4), (2176, 4), (2048, 2)):
        wide_idx = torch.arange(wide_k) // 128
        label = f"bfloat16 M={wide_m} K={wide_k} pipeline"
        act, qweight, qzeros, scales, g, ref = make_case(
            wide_m,
            64,
            wide_k,
            torch.bfloat16,
            device,
            wide_idx,
            1,
            seed=3000 + wide_k + wide_m,
        )
        out = gptq_gemm(act, qweight, qzeros, scales, g, False, 4, 1)
        torch.xpu.synchronize()
        hold.append((act, qweight, qzeros, scales, g, out))
        if within_tolerance(out, ref):
            passed += 1
            print(f"numeric {label} pass")
        else:
            failed += 1
            got = out.float().cpu()
            expect = ref.float().cpu()
            diff = (got - expect).abs()
            print(
                f"numeric {label} fail max={float(diff.max()):.6g} "
                f"mean={float(diff.mean()):.6g}"
            )

    act, qweight, qzeros, scales, g, _ = make_case(
        1, n, k, torch.float16, device, contiguous, 1, seed=7
    )
    hold.append((act, qweight, qzeros, scales, g))
    try:
        gptq_gemm(act, qweight, qzeros, scales, g, False, 4, 3)
    except Exception as exc:
        if "zero_format" not in str(exc):
            failed += 1
            print(f"numeric unknown format raised without zero_format: {exc}")
        else:
            passed += 1
            print("numeric unknown format raised")
    else:
        failed += 1
        print("numeric unknown format did not raise")

    try:
        gptq_gemm(act, qweight, qzeros[:1].contiguous(), scales, g, False, 4, 1)
    except Exception:
        passed += 1
        print("numeric mismatched zeros raised")
    else:
        failed += 1
        print("numeric mismatched zeros did not raise")

    # Contiguous bf16 M>=8 is the one-pass XMX GEMM. M=16 stays the first
    # launch. Round weights to bf16 so the reference matches the tensor-core
    # product. N=64 is the aligned DPAS tile; N=32 takes the SLM path.
    for m, n, k, zero_format, seed in (
        (16, 32, 256, 1, 11),
        (8, 64, 256, 1, 17),
        (8, 32, 256, 1, 18),
        (12, 64, 256, 1, 19),
        (16, 64, 256, 1, 20),
        (20, 32, 256, 1, 12),
        (32, 64, 256, 1, 13),
        (128, 128, 256, 2, 14),
        (512, 64, 256, 1, 15),
        (512, 128, 256, 1, 16),
    ):
        label = f"xmx bf16 M={m} K={k} N={n} zero_format={zero_format} seed={seed}"
        g = torch.arange(k) // 128
        act, qweight, qzeros, scales, g_idx, ref = make_case(
            m,
            n,
            k,
            torch.bfloat16,
            device,
            g,
            zero_format,
            seed=seed,
            round_weights=True,
        )
        out = gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, zero_format)
        torch.xpu.synchronize()
        hold.append((act, qweight, qzeros, scales, g_idx, out))
        if within_tolerance(out, ref):
            passed += 1
            print(f"numeric {label} pass")
        else:
            failed += 1
            got = out.float().cpu()
            expect = ref.float().cpu()
            diff = (got - expect).abs()
            print(
                f"numeric {label} fail max={float(diff.max()):.6g} "
                f"mean={float(diff.mean()):.6g} "
                f"got={got.flatten()[:4].tolist()} ref={expect.flatten()[:4].tolist()}"
            )

    for m, n, k, dtype in (
        (1, 128, 5120, torch.bfloat16),
        (2, 64, 17, torch.bfloat16),
        (4, 128, 5120, torch.bfloat16),
        (4, 32, 17, torch.bfloat16),
        (1, 32, 256, torch.float16),
    ):
        hidden = torch.randn(m, k, device=device, dtype=dtype)
        weight = torch.randn(n, k, device=device, dtype=dtype)
        got = torch.ops.sgl_kernel.lm_head_gemv(hidden, weight)
        ref = torch.matmul(hidden.float(), weight.float().T).to(dtype)
        torch.xpu.synchronize()
        label = f"lm_head {dtype} M={m} K={k} N={n}"
        if within_tolerance(got, ref):
            passed += 1
            print(f"numeric {label} pass")
        else:
            failed += 1
            diff = (got.float().cpu() - ref.float().cpu()).abs()
            print(f"numeric {label} fail max={float(diff.max()):.6g} mean={float(diff.mean()):.6g}")

    print(f"numeric probes passed: {passed} failed: {failed}")
    if failed:
        raise SystemExit(1)


def kernel_bench() -> None:
    import torch

    require_xpu()
    from sgl_kernel import gptq_gemm

    device = torch.device("xpu")
    shapes = (
        (5120, 17408),
        (17408, 5120),
        (5120, 12288),
        (5120, 5120),
        (256, 32),
    )
    for pack in (False, True):
        if pack:
            os.environ["SGL_GPTQ_PACK"] = "1"
        else:
            os.environ.pop("SGL_GPTQ_PACK", None)
        for k, n in shapes:
            act = torch.randn(1, k, device=device, dtype=torch.bfloat16)
            qweight = torch.randint(-(2**31), 2**31 - 1, (k // 8, n), dtype=torch.int32, device=device)
            qzeros = torch.randint(0, 15, (k // 128, n // 8), dtype=torch.int32, device=device)
            scales = torch.randn(k // 128, n, device=device, dtype=torch.bfloat16)
            g_idx = (torch.arange(k, device=device) // 128).to(torch.int32)
            for _ in range(3):
                gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, 1)
            torch.xpu.synchronize()
            started = time.perf_counter()
            iters = 10
            for _ in range(iters):
                gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, 1)
            torch.xpu.synchronize()
            dt = (time.perf_counter() - started) / iters
            weight_bytes = k * n / 2
            print(
                f"SGL_GPTQ_KERNEL pack={int(pack)} K={k} N={n} ms={dt * 1e3:.3f} "
                f"GBps={weight_bytes / dt / 1e9:.1f}",
                flush=True,
            )


def collapsed(text: str) -> bool:
    compact = "".join(text.split())
    return len(compact) < 8 or len(set(compact)) <= 1


def generation_record(stage: str) -> dict:
    require_greedy()
    if not os.path.isdir(MODEL_DIR) or not os.access(MODEL_DIR, os.R_OK):
        print(MODEL_DIR, file=sys.stderr)
        raise SystemExit(1)
    require_xpu()
    import torch
    from sglang import Engine

    # The baseline bench keeps the one-load-per-K path. Every other stage
    # loads each packed int32 once per eight K rows.
    if stage == "baseline":
        os.environ["SGL_GPTQ_SCALAR"] = "1"
    else:
        os.environ.pop("SGL_GPTQ_SCALAR", None)

    profile = os.environ.get("SGL_GPTQ_PROFILE", "") == "1"
    gemm_seconds = []
    if profile:
        import sgl_kernel.gptq_loader as loader

        original = loader.gptq_gemm

        def timed_gptq_gemm(*args, **kwargs):
            torch.xpu.synchronize()
            started = time.perf_counter()
            out = original(*args, **kwargs)
            torch.xpu.synchronize()
            gemm_seconds.append((time.perf_counter() - started, int(args[0].shape[0]), int(args[0].shape[1]), int(args[1].shape[1])))
            return out

        loader.gptq_gemm = timed_gptq_gemm
        import sgl_kernel

        sgl_kernel.gptq_gemm = timed_gptq_gemm

    prompt = os.environ.get("SGL_GPTQ_PROMPT", PROMPT)
    # Page size > 1 selects the extra_buffer GDN checkpoints. The fused
    # kernel still owns the live state, and the paged attention kernel
    # captures decode without a boolean page-table index.
    engine_kwargs = dict(
        model_path=MODEL_DIR,
        device="xpu",
        trust_remote_code=True,
        dtype="bfloat16",
        attention_backend="intel_xpu",
        linear_attn_backend="intel_xpu",
        page_size=PAGE_SIZE,
        context_length=CONTEXT_LENGTH,
    )
    if PAGE_SIZE > 1:
        engine_kwargs["mamba_radix_cache_strategy"] = "extra_buffer"
    engine = Engine(
        **engine_kwargs,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
        mem_fraction_static=0.85,
        skip_server_warmup=True,
        # XPU leaves the decode graph off unless asked. One captured batch
        # covers this greedy generate and removes the per-kernel launch tax.
        cuda_graph_backend_decode="full",
        cuda_graph_bs_decode=[1],
    )
    # The first decode compiles the attention kernel and the XPU graph.
    # One token is not enough; that compile otherwise lands inside the timer.
    engine.generate(prompt, {"temperature": 0, "max_new_tokens": 3})
    started = time.perf_counter()
    output = engine.generate(
        prompt,
        {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
    )
    elapsed = max(time.perf_counter() - started, 1e-6)
    if profile and gemm_seconds:
        total = sum(item[0] for item in gemm_seconds)
        decode = [item for item in gemm_seconds if item[1] == 1]
        prefill = [item for item in gemm_seconds if item[1] != 1]
        print(
            f"SGL_GPTQ_PROFILE calls={len(gemm_seconds)} gemm_s={total:.3f} "
            f"decode_calls={len(decode)} decode_s={sum(item[0] for item in decode):.3f} "
            f"prefill_calls={len(prefill)} prefill_s={sum(item[0] for item in prefill):.3f} "
            f"generate_s={elapsed:.3f}",
            flush=True,
        )
    if isinstance(output, list):
        output = output[0]
    text = output["text"] if isinstance(output, dict) else str(output)
    return {
        "stage": stage,
        "model": MODEL_DIR,
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "page_size": PAGE_SIZE,
        "context_length": CONTEXT_LENGTH,
        "text": text,
        "tokens_per_second": MAX_NEW_TOKENS / elapsed,
        "node": "hal",
    }


def emit_record(record: dict) -> None:
    print(RECORD_PREFIX + " " + json.dumps(record, sort_keys=True))


def generate() -> None:
    record = generation_record(os.environ.get("STAGE", "generate"))
    emit_record(record)
    if collapsed(record["text"]):
        fail(f"generation collapsed to {record['text']!r}")


def batch_report(base: str) -> None:
    import threading
    import urllib.request

    widths = [
        int(part)
        for part in os.environ.get("SGL_GPTQ_BATCH_WIDTHS", "1,2,4").split(",")
        if part.strip()
    ]
    new_tokens = int(os.environ.get("SGL_GPTQ_BATCH_TOKENS", "64"))

    def one(index: int) -> tuple[int, float]:
        body = json.dumps(
            {
                "model": MODEL_DIR,
                "prompt": PROMPT,
                "max_tokens": new_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            base + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)
        elapsed = time.perf_counter() - started
        text = payload["choices"][0]["text"]
        if collapsed(text):
            fail(f"batch request {index} collapsed to {text!r}")
        tokens = int(payload.get("usage", {}).get("completion_tokens", new_tokens))
        return tokens, elapsed

    for width in widths:
        results: list[tuple[int, float] | None] = [None] * width
        errors: list[BaseException] = []

        def run(index: int) -> None:
            try:
                results[index] = one(index)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        started = time.perf_counter()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(width)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
        if errors:
            fail(f"batch width {width} failed: {errors[0]}")
        tokens = [item[0] for item in results if item is not None]
        per = [item[0] / item[1] for item in results if item is not None and item[1] > 0]
        total = sum(tokens)
        print(
            "SGL_GPTQ_BATCH "
            f"width={width} tokens={total} wall={wall:.3f} "
            f"aggregate_tok_s={total / wall:.2f} "
            f"per_request_tok_s={sum(per) / len(per):.2f}",
            flush=True,
        )


def benchy() -> None:
    import subprocess
    import urllib.error
    import urllib.request

    port = os.environ.get("SGL_GPTQ_PORT", "30000")
    pp = os.environ.get("SGL_GPTQ_PP", "").strip() or "512"
    tg = os.environ.get("SGL_GPTQ_TG", "").strip() or "128"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_DIR,
        "--device",
        "xpu",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "intel_xpu",
        "--linear-attn-backend",
        "intel_xpu",
        "--page-size",
        str(PAGE_SIZE),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--mem-fraction-static",
        os.environ.get("SGL_GPTQ_MEM_FRACTION", "").strip()
        or ("0.92" if os.environ.get("SGL_GPTQ_SPEC") == "1" else "0.85"),
        "--skip-server-warmup",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-bs-decode",
        *[
            part.strip()
            for part in (os.environ.get("SGL_GPTQ_GRAPH_BS", "").strip() or "1").split(",")
            if part.strip()
        ],
    ]
    if PAGE_SIZE > 1:
        cmd.extend(["--mamba-radix-cache-strategy", "extra_buffer"])
    if os.environ.get("SGL_GPTQ_SPEC") != "1":
        max_running = os.environ.get("SGL_GPTQ_MAX_RUNNING", "").strip()
        if max_running:
            cmd.extend(["--max-running-requests", max_running])
        mamba_cache = os.environ.get("SGL_GPTQ_MAMBA_CACHE", "").strip()
        if mamba_cache:
            cmd.extend(["--max-mamba-cache-size", mamba_cache])
    if os.environ.get("SGL_GPTQ_SPEC") == "1":
        # MTP draft lives in this checkpoint. Verify must stay on the SYCL path.
        cmd.extend(
            [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-draft-model-path",
                MODEL_DIR,
                # topk=1 forces draft tokens to num_steps+1, which is the verify M.
                "--speculative-num-steps",
                str(_int_env("SGL_GPTQ_SPEC_STEPS", 3)),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(_int_env("SGL_GPTQ_SPEC_DRAFT_TOKENS", 4)),
                "--max-running-requests",
                "1",
                "--max-mamba-cache-size",
                str(_int_env("SGL_GPTQ_MAMBA_CACHE", 8)),
            ]
        )
    cmd.extend(["--host", "127.0.0.1", "--port", port])
    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    if os.environ.get("SGL_GPTQ_STEP") != "1":
        server_env.pop("SGL_GPTQ_STEP", None)
    server_env.pop("SGL_GPTQ_PROFILE", None)
    print(
        f"SGL_GPTQ_PAGE_SIZE {PAGE_SIZE} context_length {CONTEXT_LENGTH} "
        f"spec={os.environ.get('SGL_GPTQ_SPEC', '')}",
        flush=True,
    )
    server = subprocess.Popen(cmd, env=server_env)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.perf_counter() + 2400
        while time.perf_counter() < deadline:
            if server.poll() is not None:
                fail(f"sglang server exited {server.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            fail("sglang server did not become healthy")
        with urllib.request.urlopen(base + "/server_info", timeout=30) as resp:
            info = json.load(resp)
        print(
            "SGL_GPTQ_SERVER_INFO "
            + json.dumps(
                {
                    "page_size": info.get("page_size"),
                    "context_length": info.get("context_length"),
                    "mamba_radix_cache_strategy": info.get(
                        "mamba_radix_cache_strategy"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        proof_body = json.dumps(
            {
                "model": MODEL_DIR,
                "prompt": PROMPT,
                "max_tokens": 32,
                "temperature": 0,
            }
        ).encode()
        proof_req = urllib.request.Request(
            base + "/v1/completions",
            data=proof_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(proof_req, timeout=600) as resp:
            proof = json.load(resp)
        proof_text = proof["choices"][0]["text"]
        print("SGL_GPTQ_PAGE_PROOF " + json.dumps(proof_text), flush=True)
        if collapsed(proof_text):
            fail(f"generation collapsed to {proof_text!r}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "llama-benchy==0.4.0"]
        )
        version = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "llama-benchy"], text=True
        )
        for line in version.splitlines():
            if line.startswith("Version:"):
                print(line, flush=True)
        report = os.environ.get("SGL_GPTQ_BENCHY_REPORT", "/tmp/llama-benchy.md")
        subprocess.check_call(
            [
                "llama-benchy",
                "--base-url",
                base + "/v1",
                "--model",
                MODEL_DIR,
                "--tokenizer",
                MODEL_DIR,
                "--pp",
                pp,
                "--tg",
                tg,
                "--runs",
                "1",
                "--latency-mode",
                "none",
                "--skip-coherence",
                "--no-cache",
                "--format",
                "md",
                "--save-result",
                report,
            ]
        )
        print(Path(report).read_text())
        if os.environ.get("SGL_GPTQ_BATCH") == "1":
            batch_report(base)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def bench() -> None:
    stage = os.environ.get("STAGE", "baseline")
    record = generation_record(stage)
    emit_record(record)
    if collapsed(record["text"]):
        fail(f"generation collapsed to {record['text']!r}")
    raw = os.environ.get("BASELINE_TPS", "")
    if raw == "":
        return
    baseline = float(raw)
    if not record["tokens_per_second"] > baseline:
        fail(
            f"tokens_per_second {record['tokens_per_second']} is not greater than BASELINE_TPS {baseline}"
        )


def main() -> None:
    mode = os.environ.get("MODE", "smoke")
    if mode == "smoke":
        smoke()
    elif mode == "numeric":
        numeric()
    elif mode == "generate":
        generate()
    elif mode == "bench":
        bench()
    elif mode == "kernel":
        kernel_bench()
    elif mode == "benchy":
        benchy()
    else:
        fail(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
