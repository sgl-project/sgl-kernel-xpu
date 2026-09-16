# A/B over W4A16 grouped-GEMM launch policies at one fixed shape.
#
# Answers one question that bench_moe_wna16_grouped_gemm.py cannot: at decode
# sizes, select_w4a16_policy_id() short-circuits to the 8x64 tile
# (GroupGemmW4A16Xe20.cpp: `if (avg_m <= 8) return 0;`), so no other tile is ever
# timed there. This driver forces each compiled policy in turn with
# SGL_W4A16_POLICY_ID and times each one per (policy, rows-per-expert).
#
# The override is read once per process, so the driver runs one subprocess per
# policy, and runs the whole sweep twice: pass 2 re-measures every policy after
# every other policy has run, so thermal or co-tenant drift shows up as
# disagreement between the two passes instead of as a speedup. That check earns
# its keep on this box: in the first run of this sweep a co-tenant took the
# device part-way through, and the affected policies read ~2.2x slow while the
# ones measured before it agreed with the re-run to 0.1%. Compare the passes
# before believing any row.
#
# Shapes are GPT-OSS-120B TP=4 MoE, derived from the checkpoint config
# (hidden_size=2880, intermediate_size=2880, num_local_experts=128,
# num_experts_per_tok=4) and sglang's XPU MXFP4 shard rule
# (quantization/mxfp4.py: intermediate_size_per_partition rounded up to the
# 32-element MXFP4 block -> 2880/4 = 720 -> 736):
#   gemm1  N = 2*736 = 1472, K = 2880
#   gemm2  N = 2880,         K = 736
# Experts are the 128 local experts of a TP rank (gpt-oss shards the
# intermediate dim, not the expert list), and only `topk` of them are routed at
# conc=1, so the routing vector is `rows` on 4 experts and 0 on the other 124.
#
# Weights are random 4-bit nibbles with random E8M0 exponents rather than
# quantized bf16: the kernel's cost does not depend on the values (no
# value-dependent branch, and E2M1 has no NaN/denormal), and the correctness
# reference dequantizes exactly the bytes the kernel reads. Only the routed
# experts are dequantized for the reference; the kernel never reads the rest.
#
# Run (pin to one idle device: the shell on this box inherits a co-tenant's mask,
# and a single id is also what lets the harness log that device's clock):
#   ZE_AFFINITY_MASK=6 python benchmark/bench_moe_w4a16_policy_sweep.py \
#       --out agent_space/w4a16_policy_ab/sweep.json

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path("/sgl-workspace/intel_workspace/.claude/agents/kernel-engineer")

# select_w4a16_policy_id()'s table, in policy-id order. Ids 7..13 are the small-m
# N-width sweep under test (7..10 wide, 11..12 narrow, 13 the baseline width with
# the mainloop barrier off); they are unreachable without the override.
POLICY_NAMES = [
    "m8_n64",  # 0: what decode gets today
    "m16_n64",
    "m32_n64",
    "m64_n128",
    "m64_n128_skip",
    "m64_n256",
    "m64_n256_skip",
    "m8_n128",  # 7
    "m8_n128_skip",  # 8
    "m8_n256",  # 9
    "m8_n256_skip",  # 10
    "m8_n32",  # 11
    "m8_n16",  # 12
    "m8_n64_nobar",  # 13
]

# (BlkM, BlkN, SgCountM, SgCountN) per policy id, from w4a16_launch_policy.hpp and
# the w4a16_policy_* classes it wraps. Used to report the grid geometry next to
# each timing, because that is what makes the ranking interpretable: every entry
# has SG_N = BlkN/SgCountN = 16, so the count of 16-wide subgroup tiles a shape
# decomposes into does not depend on BlkN at all.
POLICY_TILES = [
    (8, 64, 1, 4),
    (16, 64, 1, 4),
    (32, 64, 1, 4),
    (64, 128, 1, 8),
    (64, 128, 1, 8),
    (64, 256, 1, 16),
    (64, 256, 1, 16),
    (8, 128, 1, 8),
    (8, 128, 1, 8),
    (8, 256, 1, 16),
    (8, 256, 1, 16),
    (8, 32, 1, 2),
    (8, 16, 1, 1),
    (8, 64, 1, 4),
]

# GroupGemmW4A16Xe20LauncherInstance.cpp.in:58-66 -- the grid is
# sm_count * MaxThreadsPerSM / threads_per_wg persistent work-groups, and
# cutlass's Intel query_device_multiprocessor_count() is
# gpu_slices * gpu_subslices_per_slice, i.e. Xe cores (20 on a B60), not EUs.
SM_COUNT = 20
MAX_THREADS_PER_SM = 512
SUB_GROUP = 16


def _geometry(policy_id, gemm_n, routing):
    """Work-group and subgroup tile counts for one (policy, shape), and how much of
    the persistent grid they fill."""
    blk_m, blk_n, sg_m, sg_n = POLICY_TILES[policy_id]
    threads_per_wg = SUB_GROUP * sg_m * sg_n
    wgs = SM_COUNT * MAX_THREADS_PER_SM // threads_per_wg
    # grouped_gemm_xe2.hpp:109-137: n tiles come from gemm_n padded up to the
    # work-group tile, m tiles are summed over experts.
    n_tiles = -(-gemm_n // blk_n)
    m_tiles = sum(-(-r // blk_m) for r in routing)
    wg_tiles = m_tiles * n_tiles
    sg_tiles = wg_tiles * sg_m * sg_n
    return {
        "blk_m": blk_m,
        "blk_n": blk_n,
        "sg_layout": f"{sg_m}x{sg_n}",
        "threads_per_wg": threads_per_wg,
        "persistent_wgs": wgs,
        "wg_tiles": wg_tiles,
        "sg_tiles": sg_tiles,
        "resident_sgs": SM_COUNT * MAX_THREADS_PER_SM // SUB_GROUP,
        "wg_waves": wg_tiles / wgs,
        "sg_fill": sg_tiles / (SM_COUNT * MAX_THREADS_PER_SM // SUB_GROUP),
    }


# GPT-OSS-120B TP=4: (label, N, K).
GEMMS = [("gemm1", 1472, 2880), ("gemm2", 2880, 736)]
N_EXPERTS = 128
TOPK = 4
# Rows on each routed expert. 1 is the production decode point (conc=1: 4 rows
# over 4 experts). 64 is a control: the wide-N tiles are expected to win there,
# so a flat result at r=1 is a property of the shape, not of the harness.
ROWS_PER_ROUTED_EXPERT = [1, 2, 4, 8, 64]
# The routed expert ids of the TP=4 decode step in
# bench_moe_wna16_grouped_gemm.py, reused so the two benchmarks agree, plus 7
# disjoint shifts of the same set. The timed loop cycles through all 8: one call
# reads 4 experts' weights (9.0 MB for gemm1), which fits this part's 18 MB last
# level cache, so timing the same 4 experts back to back would measure a
# cache-resident GEMM that the real model -- a different layer's weights every
# call -- never gets. Cycling gives a 72 MB footprint while every individual call
# keeps the decode shape.
ROUTED_EXPERTS = (14, 49, 79, 118)
ROUTING_SETS = 8

MXFP4_BLOCK = 32
SEED = 0
# The headline timer is pipelined throughput: PIPE_ITERS launches submitted back
# to back with a single sync, repeated REPEATS times. It is the headline rather
# than the profiler because the policies differ by only a few percent at these
# sizes, and the profiler's own numbers are not that accurate here: a
# torch.profiler session on this stack spends ~550 us of host time per launch
# (8000 launches of an 80 us kernel fill a 5 s session --
# agent_space/w4a16_policy_ab/diag_long.py), so it neither keeps the GPU busy nor
# measures a decode loop's steady state, and its per-row device time lands
# anywhere from 1% below the pipelined figure to 35% above it. Back-to-back
# submission keeps the device saturated, which is what a decode step does, and
# repeats to <=0.9% across every row of this sweep (usually <=0.1%). Host submit
# is ~9-15 us against a >=29 us kernel, so the launches stay device-bound;
# prof_device_us is still recorded per row as a cross-check and to carry the
# kernel name that proves which policy ran.
REPEATS = 7
PIPE_ITERS = 2000  # multiple of ROUTING_SETS so every repeat is a whole cycle
PROF_ITERS = 200
PROF_WARMUP = 20
# Seconds of pipelined launches before the first measurement of a shape, plus one
# discarded measurement: the clock needs a few seconds of continuous load to
# settle out of its boost state.
WARMUP_SECONDS = 5.0
# bf16 output of a K<=2880 4-bit GEMM: relative L2 against the fp32 dequantized
# reference. The kernel accumulates in fp32 and rounds once, so the error is the
# bf16 output rounding (~4e-3) plus reduction order.
REL_L2_TOL = 2e-2


def _gpu_mhz():
    """Current clock of the pinned device, or 0. Recorded so a report can show the
    samples were taken at the sustained clock. xpu-smi enumerates all physical
    devices, so the mask has to be cleared for it to see one that the mask hides."""
    mask = os.environ.get("ZE_AFFINITY_MASK", "")
    if not mask.isdigit():
        return 0
    try:
        out = subprocess.run(
            ["env", "-u", "ZE_AFFINITY_MASK", "xpu-smi", "stats", "-d", mask],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return 0
    for line in out.splitlines():
        if "GPU Frequency (MHz)" in line:
            return int(line.split("|")[2].strip())
    return 0


def _routing(rows_per_expert, shift):
    routed = {e + shift for e in ROUTED_EXPERTS}
    return [rows_per_expert if e in routed else 0 for e in range(N_EXPERTS)]


# --------------------------------------------------------------------------
# worker: one policy id, one process
# --------------------------------------------------------------------------


def _worker(policy_id, out_path):
    sys.path.insert(0, str(HARNESS))
    sys.path.insert(0, str(REPO / "tests"))
    import kernelbench as kb
    import sgl_kernel  # noqa: F401 -- registers torch.ops.sgl_kernel
    import torch
    from mxfp4_utils import dequantize_mxfp4_2d

    forced = os.environ.get("SGL_W4A16_POLICY_ID")
    if forced != str(policy_id):
        raise SystemExit(
            f"SGL_W4A16_POLICY_ID={forced!r} does not match --worker {policy_id}"
        )

    def build(gemm_n, gemm_k, rows):
        torch.manual_seed(SEED)
        torch.xpu.manual_seed_all(SEED)
        routings = [_routing(rows, shift) for shift in range(ROUTING_SETS)]
        total_m = sum(routings[0])
        # Random nibbles; random E8M0 exponents in [124, 128] -> 2^-3 .. 2^1.
        packed = torch.randint(
            0, 256, (N_EXPERTS, gemm_n, gemm_k // 2), dtype=torch.uint8, device="xpu"
        ).view(torch.int8)
        scales = torch.randint(
            124,
            129,
            (N_EXPERTS, gemm_n, gemm_k // MXFP4_BLOCK),
            dtype=torch.uint8,
            device="xpu",
        )
        act = torch.empty(
            (total_m, gemm_k), dtype=torch.bfloat16, device="xpu"
        ).normal_(0, 0.01)
        return {
            "output": torch.empty(
                (total_m, gemm_n), dtype=torch.bfloat16, device="xpu"
            ),
            "act": act,
            "packed": packed,
            "scales": scales,
            "rows": [
                torch.tensor(r, dtype=torch.int32, device="xpu") for r in routings
            ],
            "routings": routings,
            "total_m": total_m,
            "cursor": 0,
        }

    def run(inp):
        # The registered op, not the Python wrapper: the wrapper is not the subject.
        rows = inp["rows"][inp["cursor"] % ROUTING_SETS]
        inp["cursor"] += 1
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16.default(
            inp["output"],
            inp["act"],
            inp["packed"],
            inp["scales"],
            None,  # zeros
            None,  # bias
            rows,
            N_EXPERTS,
            False,  # is_int4
            MXFP4_BLOCK,
        )

    def reference(inp, routing):
        """fp32 reference over the routed experts only."""
        out = torch.empty_like(inp["output"], dtype=torch.float32)
        cursor = 0
        for expert, n_rows in enumerate(routing):
            if n_rows == 0:
                continue
            w_dq = dequantize_mxfp4_2d(
                inp["packed"][expert].view(torch.uint8),
                inp["scales"][expert],
                dtype=torch.float32,
            )
            a = inp["act"][cursor : cursor + n_rows].float()
            out[cursor : cursor + n_rows] = a @ w_dq.transpose(0, 1)
            cursor += n_rows
        return out

    import time

    def pipe_us(fn, iters):
        """Mean us/launch over `iters` back-to-back launches with one sync.

        No per-iteration sync and no profiler, so the GPU stays saturated for the
        whole measurement -- see the REPEATS comment for why that is the only
        reproducible regime on this stack. Host submit is an order of magnitude
        below the kernel, so this is device-bound; `host_us` in each row is the
        check on that.
        """
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) * 1e6 / iters

    def warm(fn, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pipe_us(fn, 1000)

    results = []
    for label, gemm_n, gemm_k in GEMMS:
        for rows in ROWS_PER_ROUTED_EXPERT:
            inp = build(gemm_n, gemm_k, rows)
            # Correctness on every routing set, so a policy is gated on all the
            # expert offsets the timed loop will use.
            err = 0.0
            for set_idx in range(ROUTING_SETS):
                inp["cursor"] = set_idx
                run(inp)
                torch.xpu.synchronize()
                ref = reference(inp, inp["routings"][set_idx])
                got = inp["output"].float()
                err = max(err, (got - ref).norm().item() / max(ref.norm().item(), 1e-6))
                del ref, got
            inp["cursor"] = 0

            fn = lambda: run(inp)  # noqa: E731
            warm(fn, WARMUP_SECONDS)
            pipe_us(fn, PIPE_ITERS)  # discarded
            samples = [pipe_us(fn, PIPE_ITERS) for _ in range(REPEATS)]
            stats = kb._stats(samples)
            # Secondary, and the source of the kernel-identity proof. Not the
            # headline: a profiled session leaves the GPU mostly idle, so this
            # reads low and unstably (see the REPEATS comment).
            kernels = kb.device_time(
                fn, iters=PROF_ITERS, warmup=PROF_WARMUP, per_kernel=True
            )
            host = kb.host_submit(fn)
            mhz = _gpu_mhz()

            results.append(
                {
                    "policy_id": policy_id,
                    "policy": POLICY_NAMES[policy_id],
                    "gemm": label,
                    "gemm_n": gemm_n,
                    "gemm_k": gemm_k,
                    "rows_per_routed_expert": rows,
                    "total_m": inp["total_m"],
                    "pipe_us_p50": stats["p50"],
                    "pipe_us_p10": stats["p10"],
                    "pipe_us_p90": stats["p90"],
                    "pipe_us_samples": samples,
                    "repeats": REPEATS,
                    "pipe_iters": PIPE_ITERS,
                    "prof_iters": PROF_ITERS,
                    "prof_device_us": kernels["total_us"],
                    "host_us": host["host_us"],
                    "gpu_mhz_after": mhz,
                    **_geometry(policy_id, gemm_n, inp["routings"][0]),
                    "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", ""),
                    "rel_l2_err": err,
                    "correct": err < REL_L2_TOL,
                    # Proof of which template ran: the mangled kernel name carries
                    # the policy type. Truncated -- the full name is huge.
                    "kernel_names": [k[:400] for k in kernels.get("kernels", {})],
                }
            )
            print(
                f"[p{policy_id} {POLICY_NAMES[policy_id]:<14}] {label} r={rows:<3} "
                f"total_m={inp['total_m']:<4} pipe_p50={stats['p50']:8.2f} us "
                f"(p10={stats['p10']:.2f} p90={stats['p90']:.2f}) "
                f"prof={kernels['total_us']:.2f} host={host['host_us']:.2f} us "
                f"rel_l2={err:.2e} {mhz}MHz",
                flush=True,
            )
            del inp
            torch.xpu.empty_cache()

    Path(out_path).write_text(
        json.dumps({"env": kb.env_stamp(), "rows": results}, indent=2)
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _driver(out_path, policy_ids, passes):
    scratch = Path(out_path).parent
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for pass_no in range(1, passes + 1):
        for policy_id in policy_ids:
            worker_out = scratch / f"worker_p{policy_id}_pass{pass_no}.json"
            env = dict(os.environ, SGL_W4A16_POLICY_ID=str(policy_id))
            print(
                f"\n=== pass {pass_no}: policy {policy_id} ({POLICY_NAMES[policy_id]}) ===",
                flush=True,
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    str(policy_id),
                    "--out",
                    str(worker_out),
                ],
                env=env,
            )
            if proc.returncode != 0:
                print(
                    f"!! policy {policy_id} pass {pass_no} failed (rc={proc.returncode})",
                    flush=True,
                )
                continue
            blob = json.loads(worker_out.read_text())
            for row in blob["rows"]:
                row["pass"] = pass_no
            rows.extend(blob["rows"])
            Path(out_path).write_text(json.dumps({"rows": rows}, indent=2))

    _print_tables(rows)
    print(f"\nraw results: {out_path}")


def _print_geometry(rows):
    """Grid geometry, so the timing ranking can be read against the decomposition."""
    for label, gemm_n, _ in GEMMS:
        print(
            f"\n### {label} (N={gemm_n}) grid geometry: WG tiles (waves of the persistent grid) / subgroup fill"
        )
        print(
            f"| policy                  |    tile | thr/WG | WGs | "
            + " | ".join(f"r={r}".rjust(20) for r in ROWS_PER_ROUTED_EXPERT)
            + " |"
        )
        for policy_id in sorted({r["policy_id"] for r in rows}):
            cells = []
            geo = None
            for r in ROWS_PER_ROUTED_EXPERT:
                hit = [
                    x
                    for x in rows
                    if x["gemm"] == label
                    and x["rows_per_routed_expert"] == r
                    and x["policy_id"] == policy_id
                ]
                if not hit:
                    cells.append("-".rjust(20))
                    continue
                geo = hit[0]
                cells.append(
                    f"{geo['wg_tiles']:4d} ({geo['wg_waves']:.2f}w) {geo['sg_fill']*100:4.0f}%".rjust(
                        20
                    )
                )
            if geo is None:
                continue
            tile = f"{geo['blk_m']}x{geo['blk_n']}"
            print(
                f"| {policy_id:>2} {POLICY_NAMES[policy_id]:<20} | {tile:>7} | {geo['threads_per_wg']:6d} | "
                f"{geo['persistent_wgs']:3d} | " + " | ".join(cells) + " |"
            )


def _print_tables(rows):
    _print_geometry(rows)
    for pass_no in sorted({r["pass"] for r in rows}):
        for label, gemm_n, gemm_k in GEMMS:
            print(
                f"\n### pass {pass_no} {label} (N={gemm_n}, K={gemm_k}) pipelined us/launch, p50 of {REPEATS}"
            )
            header = (
                "| policy                  | "
                + " | ".join(
                    f"r={r} (m={r * TOPK})".rjust(14) for r in ROWS_PER_ROUTED_EXPERT
                )
                + " |"
            )
            print(header)
            print("|" + "-" * (len(header) - 2) + "|")
            base = {}
            for r in ROWS_PER_ROUTED_EXPERT:
                hit = [
                    x
                    for x in rows
                    if x["pass"] == pass_no
                    and x["gemm"] == label
                    and x["rows_per_routed_expert"] == r
                    and x["policy_id"] == 0
                ]
                if hit:
                    base[r] = hit[0]["pipe_us_p50"]
            for policy_id in sorted({r["policy_id"] for r in rows}):
                cells = []
                for r in ROWS_PER_ROUTED_EXPERT:
                    hit = [
                        x
                        for x in rows
                        if x["pass"] == pass_no
                        and x["gemm"] == label
                        and x["rows_per_routed_expert"] == r
                        and x["policy_id"] == policy_id
                    ]
                    if not hit:
                        cells.append("-".rjust(14))
                        continue
                    v = hit[0]["pipe_us_p50"]
                    speedup = base.get(r, float("nan")) / v
                    flag = "" if hit[0]["correct"] else "!"
                    cells.append(f"{v:8.2f} ({speedup:.2f}x){flag}".rjust(14))
                print(
                    f"| {policy_id:>2} {POLICY_NAMES[policy_id]:<20} | "
                    + " | ".join(cells)
                    + " |"
                )
    bad = [r for r in rows if not r["correct"]]
    if bad:
        print(
            f"\n!! {len(bad)} rows FAILED the correctness gate (rel_l2 >= {REL_L2_TOL})"
        )
        for r in bad:
            print(
                f"   p{r['policy_id']} {r['gemm']} r={r['rows_per_routed_expert']}: rel_l2={r['rel_l2_err']:.3e}"
            )
    else:
        print(
            f"\ncorrectness: all {len(rows)} rows within rel_l2 < {REL_L2_TOL} of the fp32 dequant reference"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--worker", type=int, default=None, help="internal: measure one policy id"
    )
    ap.add_argument("--out", default="agent_space/w4a16_policy_ab/sweep.json")
    ap.add_argument(
        "--policies",
        default="all",
        help="comma-separated policy ids, or 'all', or 'decode' for the small-m band",
    )
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    if args.worker is not None:
        _worker(args.worker, args.out)
    else:
        if args.policies == "all":
            ids = list(range(len(POLICY_NAMES)))
        elif args.policies == "decode":
            ids = [0, 7, 8, 9, 10, 11, 12, 13]
        else:
            ids = [int(x) for x in args.policies.split(",")]
        _driver(args.out, ids, args.passes)
