#!/usr/bin/env python3
"""Heuristic scanner for the sgl-kernel-xpu op coding rules (see SKILL.md).

Flags lines in op code (src/, include/, python/) that match patterns
forbidden by OP-1 (host-device sync), OP-2 (global queue / non-pool queue),
OP-3 (Device API; warning), OP-5 (raw device allocation), OP-6 (device state
kept across calls), OP-7 (unit-scale defaults and downcasts; warnings) and
TLA-1 (includes from sycl-tla applications/ or examples/). OP-4 (layouts)
cannot be detected statically; review it by hand.

The rules apply to new and changed code: use --diff as the PR check. --all
reports existing code too, which predates the rules.

Hits are candidates: confirm each one in context.

Usage (from anywhere in the repo):
  scan_op_rules.py --diff [BASE]   added/changed lines since merge-base with
                                   BASE (default origin/main), including
                                   uncommitted and untracked files
  scan_op_rules.py --all           every file under src/, include/, python/
  scan_op_rules.py FILE...         whole files

Exit status is 1 if any error-level hit is found.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ("src", "include", "python")
CPP = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".inl"}
PY = {".py"}

# (rule, severity, languages, pattern, message[, exclude pattern])
RULES = [
    (
        "OP-1",
        "error",
        CPP | PY,
        r"\.item\s*(<[^>]*>)?\s*\(",
        ".item() reads a device value on the host (sync)",
    ),
    ("OP-1", "error", CPP | PY, r"\.cpu\s*\(", ".cpu() copies to the host (sync)"),
    (
        "OP-1",
        "error",
        CPP,
        r"\.to\s*\(\s*(at::|torch::|c10::)?(kCPU|Device\s*\(\s*\"cpu)",
        "copy to CPU (sync)",
    ),
    (
        "OP-1",
        "error",
        PY,
        r"\.to\s*\(\s*(device\s*=\s*)?['\"]cpu['\"]",
        "copy to CPU (sync)",
    ),
    ("OP-1", "error", PY, r"\.(tolist|numpy)\s*\(", "host read of device data (sync)"),
    (
        "OP-1",
        "error",
        CPP | PY,
        r"\.(wait|wait_and_throw)\s*\(\s*\)",
        "host blocks on the device",
    ),
    ("OP-1", "error", CPP | PY, r"\bsynchronize\s*\(", "host blocks on the device"),
    ("OP-1", "error", CPP, r"\bcopy_to_host\s*\(", "D2H copy (sync)"),
    (
        "OP-2",
        "error",
        CPP,
        r"\b(compat|dpct)::get_(default|current)_queue\b|\bget_default_queue\s*\(",
        "global queue; use c10::xpu::getCurrentXPUStream().queue()",
    ),
    (
        "OP-2",
        "error",
        CPP,
        r"\bsycl::queue\s+\w+\s*[({;]",
        "new sycl::queue; side queues come from c10::xpu::getStreamFromPool()",
    ),
    (
        "OP-5",
        "error",
        CPP,
        r"\b(sycl|compat|dpct)::(malloc\w*|aligned_alloc\w*|free)\s*[<(]|\bmalloc_(device|shared|host)\s*[<(]",
        "raw device allocation; use at::empty / at::zeros",
    ),
    (
        "OP-5",
        "error",
        CPP,
        r"\bDeviceAllocation\s*<|\bdevice_memory::allocate\b",
        "CUTLASS device allocation; use at::empty / at::zeros",
    ),
    # A static/thread_local variable (not a function) whose type holds a Tensor.
    (
        "OP-6",
        "error",
        CPP,
        r"^\s*(?:(?:inline|const|constexpr)\s+)*(?:static|thread_local)\b[^(]*\b(?:at|torch)::Tensor\b[^(]*[;={]",
        "device tensor kept across calls (static/thread_local)",
    ),
    (
        "OP-6",
        "error",
        CPP,
        r"^\s*(static|thread_local)\b.*\bdata_ptr\b",
        "device pointer kept across calls",
    ),
    (
        "OP-6",
        "warn",
        CPP,
        r"^\s*(static|thread_local)\b[^(]*\bvoid\s*\*\s*\w+\s*(=|;)",
        "pointer kept across calls; must not point to device memory",
    ),
    (
        "OP-3",
        "warn",
        CPP,
        r"\bcutlass::(\w+::)*device::\w+|\bGemmUniversalAdapter\b",
        "SYCL-TLA Device API is discouraged; give a reason in the PR, run it on "
        "the current queue",
    ),
    (
        "OP-7",
        "warn",
        CPP,
        r"\b\w*scale\w*\s*\.\s*value_or\s*\(\s*1(\.0*)?f?\s*\)",
        "missing scale defaults to 1.0; FP8 operands need user-provided scales",
    ),
    (
        "OP-7",
        "warn",
        PY,
        r"\b(?!(?:softmax|sm|attn|logit)_scale\w*\b)\w*_scale\w*\s*(:[^=,)]*)?=\s*1(\.0*)?\s*[,)]",
        "scale argument defaults to 1.0; FP8 operands need user-provided scales",
    ),
    (
        "OP-7",
        "warn",
        CPP,
        r"\.to\s*\(\s*(at::|torch::|c10::)?(ScalarType::)?k(Float8_\w+|BFloat16|Half)\b",
        "cast to a lower precision; inputs are never downcast unless the API opts in",
    ),
    (
        "OP-7",
        "warn",
        PY,
        r"\.to\s*\(\s*(dtype\s*=\s*)?torch\.(float8_\w+|bfloat16|float16|half)\b",
        "cast to a lower precision; inputs are never downcast unless the API opts in",
    ),
]
COMPILED = [
    (rule, sev, langs, re.compile(pat), msg, re.compile(extra[0]) if extra else None)
    for rule, sev, langs, pat, msg, *extra in RULES
]
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')


def tla_include_violation(inc: str) -> bool:
    """True if `inc` points into a sycl-tla applications/ or examples/ tree.

    sycl-tla is fetched at build time (CMakeLists.txt), and only its include/
    and tools/util/include/ directories are on the include path, so the
    check is on the include path itself.
    """
    return inc.startswith(("applications/", "examples/")) or bool(
        re.search(r"(^|/)sycl-tla/(applications|examples)/", inc)
    )


def strip_comment(line: str, is_py: bool) -> str:
    s = line.strip()
    if is_py:
        return "" if s.startswith("#") else line.split(" #", 1)[0]
    if s.startswith(("//", "/*", "*")):
        return ""
    return line.split("//", 1)[0]


def scan_lines(path: Path, lines, hits: list) -> None:
    ext = path.suffix
    if ext not in CPP | PY:
        return
    is_py = ext in PY
    rel = path.relative_to(ROOT) if path.is_absolute() else path
    for lineno, raw in lines:
        m = INCLUDE_RE.match(raw) if not is_py else None
        if m:
            if tla_include_violation(m.group(1)):
                hits.append(
                    (
                        rel,
                        lineno,
                        "TLA-1",
                        "error",
                        "includes kernel code from sycl-tla applications/ "
                        "or examples/; use public include/ APIs",
                        raw,
                    )
                )
            continue
        code = strip_comment(raw, is_py)
        if not code.strip():
            continue
        seen = set()
        for rule, sev, langs, pat, msg, exclude in COMPILED:
            if ext not in langs or rule in seen:
                continue
            if pat.search(code) and not (exclude and exclude.search(code)):
                seen.add(rule)
                hits.append((rel, lineno, rule, sev, msg, raw))


def whole_file(path: Path):
    text = path.read_text(errors="replace").splitlines()
    return list(enumerate(text, 1))


def git(*args) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout


def diff_lines(base: str):
    """Yield (path, [(lineno, text)]) for lines added since merge-base."""
    mb = git("merge-base", base, "HEAD").strip()
    out = git("diff", "-U0", "--no-color", "--diff-filter=AMR", mb, "--", *SCOPE)
    files, cur, lineno = {}, None, 0
    for line in out.splitlines():
        if line.startswith("+++ "):
            cur = None if line.endswith("/dev/null") else Path(line[6:])
            if cur is not None:
                files.setdefault(cur, [])
        elif line.startswith("@@") and cur is not None:
            lineno = int(re.search(r"\+(\d+)", line).group(1))
        elif line.startswith("+") and cur is not None:
            files[cur].append((lineno, line[1:]))
            lineno += 1
    for f in git("ls-files", "--others", "--exclude-standard", "--", *SCOPE).split():
        files[Path(f)] = whole_file(ROOT / f)
    return files.items()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--diff", nargs="?", const="origin/main", metavar="BASE")
    g.add_argument("--all", action="store_true")
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()

    hits = []
    if args.files:
        for f in args.files:
            p = f.resolve()
            scan_lines(p, whole_file(p), hits)
    elif args.all:
        for d in SCOPE:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    scan_lines(p, whole_file(p), hits)
    else:
        for rel, lines in diff_lines(args.diff or "origin/main"):
            scan_lines(ROOT / rel, lines, hits)

    for rel, lineno, rule, sev, msg, raw in hits:
        print(f"{rel}:{lineno}: [{rule}] {sev}: {msg}\n    {raw.strip()}")
    n_err = sum(1 for h in hits if h[3] == "error")
    print(f"{len(hits)} hit(s), {n_err} error(s), {len(hits) - n_err} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
