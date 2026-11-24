#!/usr/bin/env python3
"""Heuristic scanner for int32 stride / offset overflow in src/ and include/ (see SKILL.md).

Checks (C/C++ only):
  SO-1 error  tensor stride stored in / cast to a 32-bit integer
  SO-1 warn   numel() in a 32-bit integer; 32-bit variable / field /
              parameter whose name says stride
  SO-2 warn   outer index (block/page/slot/batch/token/seq) * stride with no
              64-bit cast on the line; confirm one operand is int64

Hits are candidates: confirm each one in context (a narrow inner stride with
a documented and guarded bound is fine, SO-4). The rules apply to new and
changed code: use --diff as the PR check.

Usage (from anywhere in the repo):
  scan_stride_overflow.py --diff [BASE]   lines added since merge-base with
                                          BASE (default origin/main),
                                          including uncommitted and untracked
  scan_stride_overflow.py --all           every C/C++ file under src/, include/
  scan_stride_overflow.py FILE...         whole files

Exit status is 1 if any error-level hit is found.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ("src", "include")
CPP = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".inl"}

NARROW = r"(?:int|int32_t|uint32_t|unsigned(?:\s+int)?|short|int16_t|uint16_t)"
WIDE_CAST = re.compile(
    r"\b(?:int64_t|uint64_t|size_t|ptrdiff_t|long\s+long|index_t)\b|\bint64\b"
)

RULES = [
    (
        "SO-1",
        "error",
        re.compile(
            r"^\s*(?:const\s+)?" + NARROW + r"\s+\w+\s*=\s*[^;]*\.strides?\s*\("
        ),
        "tensor stride stored in a 32-bit integer; use int64_t",
    ),
    (
        "SO-1",
        "error",
        re.compile(r"static_cast<\s*" + NARROW + r"\s*>\s*\([^;,]*\.stride\s*\("),
        "tensor stride narrowed to 32 bits; keep int64_t or guard the bound",
    ),
    (
        "SO-1",
        "warn",
        re.compile(
            r"^\s*(?:const\s+)?"
            + NARROW
            + r"\s+\w+\s*=\s*[^;]*\.numel\s*\(|static_cast<\s*"
            + NARROW
            + r"\s*>\s*\([^;,]*\.numel\s*\("
        ),
        "numel() in a 32-bit integer; int64_t unless the tensor size is bounded",
    ),
    (
        "SO-1",
        "warn",
        re.compile(r"^\s*(?:const\s+)?" + NARROW + r"\s+\w*stride\w*\s*(?:[;,)=]|\[)"),
        "32-bit stride variable/field/parameter; int64_t unless bounded and "
        "guarded (SO-4)",
    ),
    (
        "SO-2",
        "warn",
        re.compile(
            r"\b\w*(?:block|page|slot|batch|bid|token|tok|seq)\w*\s*\*\s*"
            r"[\w.>-]*stride\w*",
            re.I,
        ),
        "outer index * stride; confirm one operand is int64 before the multiply",
    ),
]
SKIP = re.compile(r"\bconstexpr\b|\bstatic_assert\b")


def strip_comment(line: str) -> str:
    s = line.strip()
    if s.startswith(("//", "/*", "*")):
        return ""
    return line.split("//", 1)[0]


def scan_lines(path: Path, lines, hits: list) -> None:
    if path.suffix not in CPP:
        return
    rel = path.relative_to(ROOT) if path.is_absolute() else path
    for lineno, raw in lines:
        code = strip_comment(raw)
        if not code.strip() or SKIP.search(code):
            continue
        seen = set()
        for rule, sev, pat, msg in RULES:
            if rule in seen or not pat.search(code):
                continue
            if rule == "SO-2" and WIDE_CAST.search(code):
                continue
            seen.add(rule)
            hits.append((rel, lineno, rule, sev, msg, raw))


def whole_file(path: Path):
    return list(enumerate(path.read_text(errors="replace").splitlines(), 1))


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
