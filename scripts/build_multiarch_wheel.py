#!/usr/bin/env python3
"""Build the multi-architecture ("fat") sgl-kernel wheel in one command.

Runs one wheel build per Intel GPU architecture, then merges the results with
scripts/build_fat_wheel.py:

    bmg (Xe2)  -> dist_bmg/  -> sgl_kernel/_xe20/
    cri (Xe3P) -> dist_cri/  -> sgl_kernel/_xe35/
                                merged into dist_fat/

Usage:
    scripts/build_multiarch_wheel.py                  # build both, merge
    scripts/build_multiarch_wheel.py --install        # ... and pip install it
    scripts/build_multiarch_wheel.py --targets bmg    # single arch, no merge
    scripts/build_multiarch_wheel.py --skip-build     # merge existing wheels

Requires an activated oneAPI environment and torch importable by the current
interpreter (the build runs with --no-isolation, so torch must already be
installed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Keep in sync with SGL_ARCH_PKG_DIR in CMakeLists.txt.
TARGETS = {
    "bmg": {"dist": "dist_bmg", "backend": "_xe20", "desc": "Xe2 (BMG)"},
    "cri": {"dist": "dist_cri", "backend": "_xe35", "desc": "Xe3P (CRI)"},
}


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file():
            return path
    sys.exit(f"error: could not find pyproject.toml above {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
MERGE_SCRIPT = SCRIPT_DIR / "build_fat_wheel.py"


def log(message: str) -> None:
    print(f"[multiarch] {message}", flush=True)


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def check_torch() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        sys.exit(
            "error: torch is not importable by this interpreter, but the build runs\n"
            "       with --no-isolation and needs it. Activate the right environment\n"
            f"       (and source oneAPI setvars.sh) first.\n\n{probe.stderr.strip()}"
        )
    log(f"torch {probe.stdout.strip()} ({sys.executable})")


def build_one(target: str, build_root: Path, keep_dist: bool) -> Path:
    spec = TARGETS[target]
    dist_dir = PROJECT_ROOT / spec["dist"]
    # Both passes emit the SAME wheel filename, so a stale wheel here would make
    # the merge glob ambiguous.
    if dist_dir.exists() and not keep_dist:
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Separate CMake build dirs: DPCPP_SYCL_TARGET changes SGL_ARCH_PKG_DIR and the
    # AOT flags, which a shared cache would carry over from the previous pass.
    build_dir = build_root / target

    log(f"building {target} -> {spec['desc']} -> sgl_kernel/{spec['backend']}")
    started = time.monotonic()
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "-C",
            f"build-dir={build_dir}",
            "-C",
            f"cmake.define.DPCPP_SYCL_TARGET={target}",
            "-o",
            str(dist_dir),
        ]
    )
    log(f"{target} build took {time.monotonic() - started:.0f}s")

    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        sys.exit(
            f"error: expected exactly one wheel in {dist_dir}, found "
            f"{[w.name for w in wheels]}"
        )
    return wheels[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--targets",
        default="bmg,cri",
        help="comma-separated subset of %s (default: bmg,cri)" % ",".join(TARGETS),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "dist_fat",
        help="where to write the merged wheel (default: dist_fat)",
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "build",
        help="root for per-architecture CMake build dirs (default: build)",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="merge the wheels already present in dist_bmg/ and dist_cri/",
    )
    ap.add_argument(
        "--keep-dist",
        action="store_true",
        help="do not clear the per-architecture dist dirs before building",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        help="CMAKE_BUILD_PARALLEL_LEVEL for the builds (default: leave unset)",
    )
    ap.add_argument(
        "--install",
        action="store_true",
        help="pip install the merged wheel and print the selected backend",
    )
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        sys.exit(f"error: unknown target(s) {unknown}; choose from {list(TARGETS)}")
    if not targets:
        sys.exit("error: no targets selected")

    if args.jobs:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)

    wheels: list[Path] = []
    if args.skip_build:
        for target in targets:
            dist_dir = PROJECT_ROOT / TARGETS[target]["dist"]
            found = sorted(dist_dir.glob("*.whl"))
            if len(found) != 1:
                sys.exit(
                    f"error: --skip-build needs exactly one wheel in {dist_dir}, "
                    f"found {[w.name for w in found]}"
                )
            wheels.append(found[0])
    else:
        check_torch()
        for target in targets:
            wheels.append(build_one(target, args.build_dir, args.keep_dist))

    if len(wheels) < 2:
        log(f"single-architecture build, nothing to merge: {wheels[0]}")
        final = wheels[0]
    else:
        run([sys.executable, str(MERGE_SCRIPT), str(args.out_dir), *map(str, wheels)])
        merged = sorted(args.out_dir.glob("*.whl"))
        if len(merged) != 1:
            sys.exit(
                f"error: expected one merged wheel in {args.out_dir}, "
                f"found {[w.name for w in merged]}"
            )
        final = merged[0]

    log(f"wheel: {final}")

    if args.install:
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(final)])
        run(
            [
                sys.executable,
                "-c",
                "import sgl_kernel, sgl_kernel._arch as a; "
                "print('capability:', a.probe_capability(), '-> backend:', a.select_backend())",
            ]
        )


if __name__ == "__main__":
    main()
