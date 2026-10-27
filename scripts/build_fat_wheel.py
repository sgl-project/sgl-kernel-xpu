#!/usr/bin/env python3
"""Merge per-architecture sgl-kernel wheels into one multi-architecture wheel.

Each input wheel is a normal single-architecture build whose binaries already
live in ``sgl_kernel/_xe20/`` or ``sgl_kernel/_xe35/`` (see SGL_ARCH_PKG_DIR in
CMakeLists.txt). Merging is therefore a union of disjoint directories plus a
rebuilt RECORD -- no binary patching, and ``$ORIGIN`` RPATHs stay valid because
each backend's libraries move together.

Usage:
    build_fat_wheel.py OUT_DIR WHEEL [WHEEL ...]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ARCH_DIRS = ("_xe20", "_xe35")

# Built per-wheel but arch-neutral by construction: pure host code that only
# queries device info. Copies differ in GNU build-id and in the AOT target
# triple recorded in the (kernel-free) offload bundle, so any one of them works.
ARCH_NEUTRAL_BINARIES = frozenset({"sgl_kernel/libsgl_arch_probe.so"})


def _die(msg: str) -> "NoReturn":  # noqa: F821
    sys.exit(f"error: {msg}")


def _record_hash(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return f"sha256={digest}", len(data)


def _extract(wheel: Path, dest: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(dest)


def _arch_dirs_in(tree: Path) -> list[str]:
    pkg = tree / "sgl_kernel"
    if not pkg.is_dir():
        _die(f"{tree}: no sgl_kernel/ package found")
    # Every wheel ships the stub __init__.py for both backends, so a backend only
    # counts as present when its compiled payload is there too.
    return [d for d in ARCH_DIRS if any((pkg / d).glob("*.so"))]


def _merge_tree(src: Path, dst: Path, taken: dict[str, Path], wheel_name: str) -> None:
    """Copy src into dst, refusing to silently overwrite differing files."""
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        # RECORD lists each wheel's own payload; _rewrite_record() regenerates it.
        if item.parent.name.endswith(".dist-info") and item.name.startswith("RECORD"):
            continue
        target = dst / rel
        key = rel.as_posix()
        if target.exists():
            # Identical files (pure-python sources, headers) are expected in
            # every wheel; differing ones mean the layout assumption is wrong.
            if (
                key not in ARCH_NEUTRAL_BINARIES
                and _record_hash(item)[0] != _record_hash(target)[0]
            ):
                _die(
                    f"{wheel_name}: {rel} differs from the copy already taken from "
                    f"{taken.get(key, '(unknown)')}. Architecture-specific files must live "
                    f"under sgl_kernel/<arch>/."
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        taken[key] = wheel_name


def _rewrite_record(tree: Path) -> Path:
    dist_infos = sorted(tree.glob("*.dist-info"))
    if len(dist_infos) != 1:
        _die(f"expected exactly one .dist-info, found {[d.name for d in dist_infos]}")
    dist_info = dist_infos[0]
    record = dist_info / "RECORD"

    rows = []
    for path in sorted(tree.rglob("*")):
        if path.is_dir() or path == record:
            continue
        digest, size = _record_hash(path)
        rows.append([str(path.relative_to(tree)), digest, size])
    rows.append([str(record.relative_to(tree)), "", ""])

    with record.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return dist_info


def _wheel_filename(dist_info: Path) -> str:
    """Rebuild the canonical wheel name, independent of input argument order."""
    name_version = dist_info.name[: -len(".dist-info")]
    tags = [
        line.split(":", 1)[1].strip()
        for line in (dist_info / "WHEEL").read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag:")
    ]
    if not tags:
        _die(f"{dist_info.name}/WHEEL has no Tag: entry")
    return f"{name_version}-{tags[0]}.whl"


def _repack(tree: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(tree))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("wheels", type=Path, nargs="+")
    args = ap.parse_args()

    if len(args.wheels) < 2:
        _die("need at least two wheels to merge")
    for wheel in args.wheels:
        if not wheel.is_file():
            _die(f"{wheel}: not found")

    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / "merged"
        merged.mkdir()
        taken: dict[str, Path] = {}
        seen_arches: list[str] = []

        for idx, wheel in enumerate(args.wheels):
            # Input wheels usually share the same filename; index keeps them apart.
            staged = Path(tmp) / f"stage-{idx}-{wheel.stem}"
            _extract(wheel, staged)
            arches = _arch_dirs_in(staged)
            if not arches:
                _die(
                    f"{wheel.name}: contains no sgl_kernel/_xe* backend. It was probably built "
                    "before the per-architecture install layout; rebuild it."
                )
            for arch in arches:
                if arch in seen_arches:
                    _die(
                        f"{wheel.name}: backend {arch} already provided by an earlier wheel"
                    )
            seen_arches.extend(arches)
            print(f"  {wheel.name}: backends {arches}")
            _merge_tree(staged, merged, taken, wheel.name)

        probe = merged / "sgl_kernel" / "libsgl_arch_probe.so"
        if not probe.exists():
            _die(
                "merged wheel has no sgl_kernel/libsgl_arch_probe.so; backend selection would fail"
            )

        dist_info = _rewrite_record(merged)
        out = args.out_dir / _wheel_filename(dist_info)
        _repack(merged, out)

    print(f"\nmerged backends {sorted(seen_arches)} -> {out}")
    print(f"  dist-info: {dist_info.name}")
    print(f"  size: {out.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
