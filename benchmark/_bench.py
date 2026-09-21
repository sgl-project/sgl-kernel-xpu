"""Standardized benchmark schema + emitter .

Every benchmark declares *what it measures* (``BenchSpec`` / ``ProblemShape``)
and calls *one emitter* (``make_row`` → ``format_tables``); the emitter owns the
format, units, rounding, comparison math, and column layout. Same table shape
for every kernel:

    kernel · arch · provider · dtype · problem_shape(n-D tuple)
          · time_us · bandwidth_gbs · tflops
          · reference_provider · reference_time_us · speedup

- Time is always ``time_us`` (µs), median — never ms or GB/s as the primary column.
- Problem size is one ``problem_shape`` column holding the raw n-D tuple; the axis
  names + description print once above the table. A 2D and a 5D kernel produce the
  same column count.
- Reference is in-row and optional: ``speedup = reference_time_us / time_us``
  (>1 ⇒ sglang faster); ``—`` when no baseline was timed.
- ``arch`` is the live device the run executed on (``current_arch()``).

"""

import functools
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch

# ── arch probe (mirror of tests/_xpu.py) ─────────────────────────────────────
KNOWN_ARCHS = frozenset({"xe20", "xe35"})


@functools.lru_cache(maxsize=1)
def current_arch() -> Optional[str]:
    """Live XPU arch tag ('xe20'|'xe35'), or None off-XPU / unknown."""
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return None
    device = torch.xpu.current_device()
    major, minor = torch.ops.sgl_kernel.query_device.default(device)
    if major == 2:
        return "xe20"
    if major == 3 and minor == 5:
        return "xe35"
    return None


# ── schema ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProblemShape:
    """The named axes of a kernel's problem size + a one-line description.

    ``axes`` labels each slot of the n-D ``problem_shape`` tuple; ``description``
    is printed once above the table (never per row)."""

    axes: Tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class BenchSpec:
    """What a benchmark measures: kernel name, problem shape, and metric set."""

    kernel: str
    shape: ProblemShape
    metrics: Tuple[str, ...] = ("time_us", "bandwidth_gbs", "tflops")


@dataclass
class Row:
    """One canonical result row (also carries its spec's shape for the header)."""

    kernel: str
    arch: Optional[str]
    provider: str
    dtype: str
    problem_shape: Tuple
    time_us: Optional[float] = None
    bandwidth_gbs: Optional[float] = None
    tflops: Optional[float] = None
    reference_provider: Optional[str] = None
    reference_time_us: Optional[float] = None
    speedup: Optional[float] = None
    _axes: Tuple[str, ...] = field(default=(), repr=False)
    _description: str = field(default="", repr=False)


# ── the one emitter ───────────────────────────────────────────────────────────
def make_row(
    spec: BenchSpec,
    *,
    dtype,
    size: Sequence,
    time_us: float,
    bandwidth_gbs: Optional[float] = None,
    tflops: Optional[float] = None,
    provider: str = "sglang",
    reference_provider: Optional[str] = None,
    reference_time_us: Optional[float] = None,
) -> Row:
    """Build a canonical row. ``arch`` is stamped from the live device and
    ``speedup`` is derived (``reference_time_us / time_us``). Metrics not declared
    in ``spec.metrics`` are dropped so the table reflects what was measured."""
    speedup = (
        reference_time_us / time_us
        if reference_time_us is not None and time_us
        else None
    )
    return Row(
        kernel=spec.kernel,
        arch=current_arch(),
        provider=provider,
        dtype=str(dtype),
        problem_shape=tuple(size),
        time_us=time_us,
        bandwidth_gbs=bandwidth_gbs if "bandwidth_gbs" in spec.metrics else None,
        tflops=tflops if "tflops" in spec.metrics else None,
        reference_provider=reference_provider,
        reference_time_us=reference_time_us,
        speedup=speedup,
        _axes=spec.shape.axes,
        _description=spec.shape.description,
    )


# Fixed column order + how each cell is rendered. Numeric columns are right-aligned.
_COLUMNS = (
    ("kernel", str, False),
    ("arch", str, False),
    ("provider", str, False),
    ("dtype", str, False),
    ("problem_shape", None, False),  # tuple rendered verbatim
    ("time_us", 2, True),
    ("bandwidth_gbs", 2, True),
    ("tflops", 2, True),
    ("reference_provider", str, False),
    ("reference_time_us", 2, True),
    ("speedup", 3, True),
)

_MISSING = "—"


def _cell(value, kind) -> str:
    if value is None:
        return _MISSING
    if kind is None:  # tuple / problem_shape
        return str(tuple(value))
    if isinstance(kind, int):  # numeric with `kind` decimal places
        return f"{value:.{kind}f}"
    return str(value)


def _markdown_table(rows: Sequence[Row]) -> str:
    header = [name for name, _, _ in _COLUMNS]
    body = [[_cell(getattr(r, name), kind) for name, kind, _ in _COLUMNS] for r in rows]

    widths = [
        max(len(header[i]), *(len(row[i]) for row in body)) if body else len(header[i])
        for i in range(len(header))
    ]
    right = [is_num for _, _, is_num in _COLUMNS]

    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(c.rjust(widths[i]) if right[i] else c.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"

    sep = (
        "| "
        + " | ".join(
            ("-" * (widths[i] - 1) + ":") if right[i] else ("-" * widths[i])
            for i in range(len(header))
        )
        + " |"
    )

    return "\n".join([fmt(header), sep, *(fmt(row) for row in body)])


def format_tables(rows: Sequence[Row]) -> str:
    """Render one standardized table per kernel (grouped, first-seen order)."""
    if not rows:
        return ""

    groups = {}
    for r in rows:
        groups.setdefault(r.kernel, []).append(r)

    blocks = []
    for kernel, group in groups.items():
        arch = current_arch()
        head = group[0]
        bar = "=" * 84
        axes = f"({', '.join(head._axes)})" if head._axes else "(…)"
        shape_line = f"problem_shape: {axes}"
        if head._description:
            shape_line += f" — {head._description}"
        blocks.append(
            "\n".join(
                [
                    bar,
                    f"{kernel}  [arch={arch}]",
                    bar,
                    shape_line,
                    _markdown_table(group),
                ]
            )
        )
    return "\n\n".join(blocks)
