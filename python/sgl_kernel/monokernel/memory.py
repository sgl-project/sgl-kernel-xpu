"""GPU memory budget of a server that runs the mono decode kernels.

SGLang sizes its pools on XPU from ``total - torch.xpu.memory_allocated()``:
memory outside torch's allocator (driver and Level Zero heaps, kernel modules,
the command lists of captured graphs, oneCCL buffers, the kernels' own scratch)
is not counted, and grows by several GiB while graphs are captured and the
first requests run. When the card is physically full the driver evicts
buffers, including the peer card's mappings, and preempts the resident layer
kernels mid-wait: the ranks stall until the wait deadline fails the step. The
budget below estimates that memory from the graph configuration, refuses a
``--mem-fraction-static`` that leaves less than ``SGLANG_XPU_MONO_VRAM_RESERVE_GIB``
(default 1.5 GiB) of physical headroom, and caps torch's allocator so that a
torch allocation past the budget fails (an out-of-memory error stops the
server) instead of pushing the card into eviction."""

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

RESERVE_ENV = "SGLANG_XPU_MONO_VRAM_RESERVE_GIB"
# 0: only warn (no refusal, no allocator cap), e.g. to test the fail-stop
CHECK_ENV = "SGLANG_XPU_MONO_VRAM_CHECK"
DEFAULT_RESERVE_GIB = 1.5
GIB = float(1 << 30)

# Memory outside torch at steady state (GiB), measured on B70 (Qwen3.6-35B-A3B
# FP8, TP2, rank 0): the driver, oneCCL and the first-use kernel modules of the
# decode path and the first requests; MTP adds the draft model's and the verify
# path's modules; then the command lists of each captured decode-phase graph
# (target decode or verify, draft decode, draft extend) and of each breakable
# prefill shape.
BASE_GIB = 2.6
SPECULATIVE_GIB = 0.5
DECODE_GRAPH_GIB = 0.03
PREFILL_SHAPE_GIB = 0.0075
DEFAULT_PREFILL_SHAPES = 50
# torch's steady use above the static pools (graph pools, activations)
ACTIVATION_GIB = 0.5
# torch's allocator stops this far below the estimated physical limit: past
# it an allocation fails (and stops the server) instead of evicting
CAP_MARGIN_GIB = 0.25


def reserve_gib() -> float:
    value = os.environ.get(RESERVE_ENV)
    gib = float(value) if value else DEFAULT_RESERVE_GIB
    if gib < 0:
        raise ValueError(f"{RESERVE_ENV}={value}: the reserve must not be negative")
    return gib


@dataclass(frozen=True)
class Graphs:
    """The graphs a server captures."""

    decode_bs: tuple  # decode / verify batch sizes (empty: decode graphs off)
    prefill_shapes: int  # breakable / full prefill token buckets (0: off)
    speculative: bool  # MTP: verify, draft decode and draft extend graphs

    @property
    def decode_graphs(self) -> int:
        return len(self.decode_bs) * (3 if self.speculative else 1)

    @property
    def prefill_graphs(self) -> int:
        return self.prefill_shapes * (2 if self.speculative else 1)


@dataclass(frozen=True)
class Budget:
    total_gib: float
    mem_fraction_static: float
    outside_torch_gib: float  # estimated memory outside torch's allocator
    reserve_gib: float

    @property
    def torch_gib(self) -> float:
        """Torch's share: SGLang fills its pools up to the static fraction,
        then graph pools and activations come on top."""
        return self.total_gib * self.mem_fraction_static + ACTIVATION_GIB

    @property
    def headroom_gib(self) -> float:
        return self.total_gib - self.torch_gib - self.outside_torch_gib

    @property
    def fits(self) -> bool:
        return self.headroom_gib >= self.reserve_gib

    @property
    def max_fraction(self) -> float:
        """The largest --mem-fraction-static that leaves the reserve."""
        free = (
            self.total_gib - self.outside_torch_gib - self.reserve_gib - ACTIVATION_GIB
        )
        return math.floor(free / self.total_gib * 100) / 100

    @property
    def torch_cap_fraction(self) -> float:
        """Torch's allocator limit (a fraction of the card): the physical limit
        less the memory outside torch (torch may use the reserve, not more)."""
        return max(
            0.0,
            min(
                1.0,
                (self.total_gib - self.outside_torch_gib - CAP_MARGIN_GIB)
                / self.total_gib,
            ),
        )


def outside_torch_gib(graphs: Graphs) -> float:
    return (
        BASE_GIB
        + (SPECULATIVE_GIB if graphs.speculative else 0.0)
        + DECODE_GRAPH_GIB * graphs.decode_graphs
        + PREFILL_SHAPE_GIB * graphs.prefill_graphs
    )


def budget(
    total_bytes: int,
    mem_fraction_static: float,
    graphs: Graphs,
    reserve: Optional[float] = None,
) -> Budget:
    return Budget(
        total_gib=total_bytes / GIB,
        mem_fraction_static=float(mem_fraction_static),
        outside_torch_gib=outside_torch_gib(graphs),
        reserve_gib=reserve_gib() if reserve is None else reserve,
    )


def refusal(b: Budget, graphs: Graphs, max_running: Optional[int]) -> str:
    fewer = []
    if len(graphs.decode_bs) > 8:  # the monokernels cover up to 8 rows
        fewer.append("--cuda-graph-max-bs-decode 8")
        if not max_running or max_running > 8:
            fewer.append("--max-running-requests 8")
    if graphs.prefill_shapes:
        fewer.append("--cuda-graph-backend-prefill disabled")
    return (
        f"--mem-fraction-static {b.mem_fraction_static:g} leaves {b.headroom_gib:.2f} GiB of physical headroom on "
        f"the {b.total_gib:.2f} GiB card: SGLang does not count torch's use above its pools (about "
        f"{ACTIVATION_GIB:g} GiB) and memory outside torch, about "
        f"{b.outside_torch_gib:.2f} GiB here (driver, oneCCL, {graphs.decode_graphs} decode-phase and "
        f"{graphs.prefill_graphs} prefill graphs), and the monokernels need {b.reserve_gib:g} GiB "
        f"({RESERVE_ENV}): a full card makes the driver evict buffers and stall them. Use "
        f"--mem-fraction-static {b.max_fraction:.2f} or less"
        + (f", or capture fewer graphs ({', '.join(fewer)})" if fewer else "")
    )


def server_graphs() -> Graphs:
    """The graphs of this server (SGLang's resolved configuration)."""
    from sglang.srt.runtime_context import get_exec, get_server_args

    graph = get_exec().graph.cuda_graph_config
    decode_bs = (
        tuple(graph.decode.bs or ()) if graph.decode.backend != "disabled" else ()
    )
    prefill = 0
    if graph.prefill.backend != "disabled":
        prefill = len(graph.prefill.bs) if graph.prefill.bs else DEFAULT_PREFILL_SHAPES
    return Graphs(
        decode_bs, prefill, get_server_args().speculative_algorithm is not None
    )


def apply(device: torch.device) -> Budget:
    """Checks this server's budget (raises when it does not leave the reserve)
    and caps torch's allocator on ``device``; run before SGLang sizes its pools."""
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    graphs = server_graphs()
    total = torch.xpu.get_device_properties(device).total_memory
    b = budget(total, args.mem_fraction_static, graphs)
    enforce = os.environ.get(CHECK_ENV, "1") != "0"
    if not b.fits:
        message = refusal(b, graphs, getattr(args, "max_running_requests", None))
        if enforce:
            raise ValueError(message)
        logger.warning("%s (%s=0: not enforced)", message, CHECK_ENV)
    if enforce:
        torch.xpu.set_per_process_memory_fraction(b.torch_cap_fraction, device)
    logger.info(
        "XPU monokernel memory: card %.2f GiB, torch pools %.2f GiB (capped at %.2f GiB), outside torch about "
        "%.2f GiB (%d decode-phase, %d prefill graphs), headroom %.2f GiB (reserve %g GiB)",
        b.total_gib,
        b.torch_gib,
        b.torch_cap_fraction * b.total_gib,
        b.outside_torch_gib,
        graphs.decode_graphs,
        graphs.prefill_graphs,
        b.headroom_gib,
        b.reserve_gib,
    )
    return b


class PressureWatch:
    """Samples the card's free memory and warns (once) when it runs low: the
    measured memory outside torch, and the fraction that would leave the reserve."""

    LOW_GIB = 0.25

    def __init__(self, device: torch.device, b: Budget, period_s: float = 2.0):
        self.device, self.budget, self.period_s = device, b, period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="xpu-monokernel-memory"
        )
        self._thread.start()

    def sample(self) -> Optional[str]:
        free, total = torch.xpu.mem_get_info(self.device)
        if free / GIB >= self.LOW_GIB:
            return None
        outside = (total - free - torch.xpu.memory_reserved(self.device)) / GIB
        fraction = (
            math.floor(
                (total / GIB - outside - self.budget.reserve_gib) / (total / GIB) * 100
            )
            / 100
        )
        return (
            f"XPU monokernel: the card has {free / GIB:.2f} GiB free; memory outside torch is "
            f"{outside:.2f} GiB (estimated {self.budget.outside_torch_gib:.2f}). A full card makes the "
            f"driver evict buffers and stall the monokernels; --mem-fraction-static {fraction:.2f} or "
            "less, or fewer captured graphs, leaves the reserve"
        )

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                message = self.sample()
            except Exception:  # noqa: BLE001 (the query is best effort)
                return
            if message:
                logger.warning(message)
                return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
