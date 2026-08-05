#!/usr/bin/env python3
"""Build a publishable sglang-kernel-xpu wheel (manylinux version).

This automates the flow:
  1. build a local wheel into dist/ with pip wheel -v . --wheel-dir dist
  2. run auditwheel once to discover vendored shared libraries
  3. parse <package>.libs entries into auditwheel --exclude names
  4. run auditwheel again with those dependencies excluded
  5. keep only the publishable payload directories and fix RECORD
  6. repack the final manylinux wheel into dist/
  7. remove intermediate files/directories, leaving the final wheel in dist/

The script intentionally shells out to the active Python/pip/auditwheel so it uses
whatever Intel/PyTorch/XPU environment is already configured by the caller. The
project root is discovered by walking upward from this file until pyproject.toml
is found; all dist/ and build commands are rooted there.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable


class BuildError(RuntimeError):
    pass


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file():
            return path
    raise BuildError(f"could not find pyproject.toml above {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DIST_DIR = PROJECT_ROOT / "dist"
WHEELHOUSE_DIR = DIST_DIR / "wheelhouse"


def log(message: str) -> None:
    print(f"[build-wheel] {message}", flush=True)


def run(cmd: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_capture(cmd: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> str:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout


INSTALL_COMMANDS = {
    "auditwheel": ["pip", "install", "auditwheel"],
    "patchelf": ["apt-get", "install", "patchelf", "-y"],
}


def ensure_tool(name: str) -> None:
    if shutil.which(name) is not None:
        return

    install_cmd = INSTALL_COMMANDS.get(name)
    if install_cmd is None:
        raise BuildError(f"required tool '{name}' was not found on PATH and no installer is configured")

    log(f"required tool '{name}' was not found; installing with: {' '.join(install_cmd)}")
    run(install_cmd)
    if shutil.which(name) is None:
        raise BuildError(f"tool '{name}' is still not available on PATH after: {' '.join(install_cmd)}")


def clean_previous_outputs() -> None:
    for path in (WHEELHOUSE_DIR, DIST_DIR / "publishable-work", DIST_DIR / "final"):
        if path.exists():
            log(f"removing stale {path.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(path)


def current_torch_lib_dir() -> Path:
    code = "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise BuildError(
            "failed to import torch and locate torch/lib in the active environment.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    torch_lib = Path(proc.stdout.strip())
    if not torch_lib.is_dir():
        raise BuildError(f"torch lib directory does not exist: {torch_lib}")
    return torch_lib


def oneapi_lib_dirs(env: dict[str, str]) -> list[Path]:
    oneapi_root = env.get("ONEAPI_ROOT")
    if not oneapi_root:
        log("ONEAPI_ROOT is not set; no oneAPI lib path added to LD_LIBRARY_PATH")
        return []

    root = Path(oneapi_root)
    candidates = sorted(root.glob("*/lib"), reverse=True)
    if (root / "lib").is_dir():
        candidates.insert(0, root / "lib")

    lib_dirs = [path for path in candidates if path.is_dir()]
    if not lib_dirs:
        log(f"no oneAPI lib directory found under ONEAPI_ROOT={oneapi_root}")
    return lib_dirs


def prepend_ld_library_path(env: dict[str, str], paths: Iterable[Path]) -> None:
    new_entries = [str(path) for path in paths]
    existing_entries = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(new_entries + existing_entries)


def auditwheel_env() -> dict[str, str]:
    env = os.environ.copy()
    torch_lib = current_torch_lib_dir()
    ld_paths = [torch_lib, *oneapi_lib_dirs(env)]
    prepend_ld_library_path(env, ld_paths)
    log(f"LD_LIBRARY_PATH includes torch lib: {torch_lib}")
    for lib_dir in ld_paths[1:]:
        log(f"LD_LIBRARY_PATH includes oneAPI lib: {lib_dir}")
    return env


def find_single_wheel(directory: Path, *, exclude: Iterable[Path] = ()) -> Path:
    excluded = {p.resolve() for p in exclude}
    wheels = sorted(
        p for p in directory.glob("*.whl") if p.resolve() not in excluded and p.is_file()
    )
    if len(wheels) != 1:
        names = "\n  ".join(str(p.relative_to(PROJECT_ROOT)) for p in wheels) or "<none>"
        raise BuildError(f"expected exactly one wheel in {directory}, found:\n  {names}")
    return wheels[0]


def build_wheel(args: argparse.Namespace) -> Path:
    DIST_DIR.mkdir(exist_ok=True)
    if args.skip_build:
        wheel = Path(args.input_wheel).resolve() if args.input_wheel else find_single_wheel(DIST_DIR)
        log(f"using existing wheel: {wheel.relative_to(PROJECT_ROOT) if wheel.is_relative_to(PROJECT_ROOT) else wheel}")
        return wheel

    for wheel in DIST_DIR.glob("*.whl"):
        log(f"removing stale input wheel {wheel.relative_to(PROJECT_ROOT)}")
        wheel.unlink()

    if args.build_command:
        run(args.build_command, env=os.environ.copy())
    else:
        run(
            [sys.executable, "-m", "pip", "wheel", "-v", ".", "--wheel-dir", "dist"],
            env=os.environ.copy(),
        )

    return find_single_wheel(DIST_DIR)


def auditwheel_repair(input_wheel: Path, excludes: list[str], env: dict[str, str]) -> Path:
    if WHEELHOUSE_DIR.exists():
        shutil.rmtree(WHEELHOUSE_DIR)
    cmd = ["auditwheel", "repair", "--strip", "--wheel-dir", str(WHEELHOUSE_DIR)]
    for lib in excludes:
        cmd.extend(["--exclude", lib])
    cmd.append(str(input_wheel))
    run(cmd, env=env)
    return find_single_wheel(WHEELHOUSE_DIR)


def strip_auditwheel_hash(lib_file_name: str) -> str:
    """Convert libfoo-abcdef12.so.1 -> libfoo.so.1 for auditwheel --exclude."""
    # auditwheel appends a hash between the logical library name and the .so
    # suffix. The logical name itself may contain hyphens, so anchor on `.so`.
    return re.sub(r"-[^-/.]+(?=\.so(?:\.|$))", "", lib_file_name)


def exclude_name_candidates(lib_file_name: str) -> list[str]:
    """Return possible external sonames for an auditwheel-renamed library.

    auditwheel's vendored filename keeps the real file version, for example
    libsycl-18fa367f.so.8.0.0 -> libsycl.so.8.0.0. Some ELF DT_NEEDED entries
    use the ABI soname instead, for example libsycl.so.8, so include that form
    as an additional --exclude candidate.
    """
    full_name = strip_auditwheel_hash(lib_file_name)
    candidates = [full_name]
    match = re.match(r"^(?P<prefix>.+\.so)\.(?P<versions>\d+(?:\.\d+)+)$", full_name)
    if match:
        abi_name = f"{match.group('prefix')}.{match.group('versions').split('.')[0]}"
        if abi_name not in candidates:
            candidates.append(abi_name)
    return candidates


def discover_excluded_libs(repaired_wheel: Path, work_dir: Path) -> list[str]:
    unpack_dir = work_dir / "first-repair-unpacked"
    if unpack_dir.exists():
        shutil.rmtree(unpack_dir)
    unpack_dir.mkdir(parents=True)
    with zipfile.ZipFile(repaired_wheel) as zf:
        zf.extractall(unpack_dir)

    libs_dirs = sorted(p for p in unpack_dir.iterdir() if p.is_dir() and p.name.endswith(".libs"))
    if not libs_dirs:
        log("no *.libs directory found in first repair; no auditwheel excludes needed")
        return []

    excludes: set[str] = set()
    for libs_dir in libs_dirs:
        for path in sorted(libs_dir.iterdir()):
            if path.is_file() and ".so" in path.name:
                excludes.update(exclude_name_candidates(path.name))

    result = sorted(excludes)
    if result:
        log("external shared libraries to exclude from final repair:")
        for lib in result:
            log(f"  --exclude {lib}")
    else:
        log("*.libs directories existed, but no shared libraries were found")
    return result


def auditwheel_hashed_lib_map(unpacked_dir: Path) -> dict[str, str]:
    """Map auditwheel vendored library names back to external sonames.

    If auditwheel still rewrites an extension to depend on a hashed vendored name
    and we later remove the *.libs directory, imports fail with errors like:
    `ImportError: libsycl-18fa367f.so.8.0.0: cannot open shared object file`.
    Before copying the final payload, patch those DT_NEEDED entries back to the
    un-hashed external soname.
    """
    mapping: dict[str, str] = {}
    for libs_dir in sorted(p for p in unpacked_dir.iterdir() if p.is_dir() and p.name.endswith(".libs")):
        for path in sorted(libs_dir.iterdir()):
            if path.is_file() and ".so" in path.name:
                mapping[path.name] = strip_auditwheel_hash(path.name)
    return mapping


def iter_patchable_elf_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.endswith(".libs") for part in path.relative_to(root).parts):
            continue
        if path.name.endswith(".so") or ".so." in path.name:
            yield path


def restore_external_needed(unpacked_dir: Path) -> None:
    hashed_to_external = auditwheel_hashed_lib_map(unpacked_dir)
    if not hashed_to_external:
        return

    replacements = 0
    for elf_path in iter_patchable_elf_files(unpacked_dir):
        needed = run_capture(["patchelf", "--print-needed", str(elf_path)]).splitlines()
        for old_name in needed:
            new_name = hashed_to_external.get(old_name)
            if new_name is None:
                continue
            run(["patchelf", "--replace-needed", old_name, new_name, str(elf_path)])
            replacements += 1
            log(
                "restored external DT_NEEDED "
                f"{old_name} -> {new_name} in {elf_path.relative_to(unpacked_dir)}"
            )

    if replacements == 0:
        log("no hashed auditwheel DT_NEEDED entries needed restoring")


def choose_payload_dirs(unpacked_dir: Path) -> tuple[Path, Path, Path]:
    include_dir = unpacked_dir / "include"
    package_dir = unpacked_dir / "sgl_kernel"
    dist_infos = sorted(unpacked_dir.glob("*.dist-info"))

    missing = [str(p.name) for p in (include_dir, package_dir) if not p.is_dir()]
    if missing:
        raise BuildError(f"final repaired wheel is missing expected payload dirs: {', '.join(missing)}")
    if len(dist_infos) != 1:
        found = ", ".join(p.name for p in dist_infos) or "<none>"
        raise BuildError(f"expected exactly one *.dist-info directory, found: {found}")
    return include_dir, package_dir, dist_infos[0]


def copy_publishable_payload(repaired_wheel: Path, work_dir: Path) -> Path:
    unpack_dir = work_dir / "final-repair-unpacked"
    final_dir = DIST_DIR / "final"
    for path in (unpack_dir, final_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    with zipfile.ZipFile(repaired_wheel) as zf:
        zf.extractall(unpack_dir)

    restore_external_needed(unpack_dir)
    include_dir, package_dir, dist_info_dir = choose_payload_dirs(unpack_dir)
    for src in (include_dir, package_dir, dist_info_dir):
        shutil.copytree(src, final_dir / src.name)
    rewrite_record(final_dir, dist_info_dir.name)
    return final_dir


def wheel_record_hash(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
    return f"sha256={digest}", str(len(data))


def should_drop_record_path(record_path: str) -> bool:
    normalized = record_path.replace("\\", "/")
    return (
        normalized.startswith("lib/")
        or normalized.startswith("test/")
        or normalized.startswith("tests/")
        or ".libs/" in normalized
        or normalized.endswith(".libs")
        or "/__pycache__/" in normalized
    )


def rewrite_record(final_dir: Path, dist_info_name: str) -> None:
    record = final_dir / dist_info_name / "RECORD"
    if not record.is_file():
        raise BuildError(f"RECORD file was not found: {record}")

    rows: list[list[str]] = []
    for file_path in sorted(p for p in final_dir.rglob("*") if p.is_file()):
        rel = file_path.relative_to(final_dir).as_posix()
        if should_drop_record_path(rel):
            continue
        if rel == f"{dist_info_name}/RECORD":
            rows.append([rel, "", ""])
        else:
            digest, size = wheel_record_hash(file_path)
            rows.append([rel, digest, size])

    with record.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    log(f"rewrote {record.relative_to(final_dir)} with {len(rows)} entries")


def repack_wheel(final_dir: Path, output_wheel: Path) -> None:
    if output_wheel.exists():
        output_wheel.unlink()
    with zipfile.ZipFile(output_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(final_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(final_dir).as_posix())
    log(f"created publishable wheel: {output_wheel}")


def final_output_path(repaired_wheel: Path, output_arg: str | None) -> Path:
    # Keep the published artifact under ./dist. If a path is provided, use its
    # basename so this script does not depend on this server's absolute paths.
    output_name = Path(output_arg).name if output_arg else repaired_wheel.name
    return DIST_DIR / output_name


def cleanup_intermediate_outputs(output_wheel: Path) -> None:
    output_wheel = output_wheel.resolve()

    for path in (WHEELHOUSE_DIR, DIST_DIR / "publishable-work", DIST_DIR / "final"):
        if path.exists():
            log(f"removing intermediate {path.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(path)

    for wheel in DIST_DIR.glob("*.whl"):
        if wheel.resolve() != output_wheel:
            log(f"removing intermediate wheel {wheel.relative_to(PROJECT_ROOT)}")
            wheel.unlink()


def verify_wheel(output_wheel: Path) -> None:
    with zipfile.ZipFile(output_wheel) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise BuildError(f"zip integrity check failed at member: {bad}")
        names = set(zf.namelist())

    required_prefixes = ("include/", "sgl_kernel/")
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in names):
            raise BuildError(f"output wheel has no entries under {prefix}")
    if not any(name.endswith(".dist-info/RECORD") for name in names):
        raise BuildError("output wheel has no dist-info RECORD")
    forbidden = [n for n in names if n.startswith(("lib/", "test/", "tests/")) or ".libs/" in n]
    if forbidden:
        preview = "\n  ".join(sorted(forbidden)[:20])
        raise BuildError(f"output wheel still contains forbidden lib/test/.libs entries:\n  {preview}")
    log("zip integrity and payload checks passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="reuse --input-wheel or the only wheel already in dist/")
    parser.add_argument("--input-wheel", help="existing wheel to repair; implies --skip-build")

    parser.add_argument(
        "--build-command",
        nargs=argparse.REMAINDER,
        help=(
            "custom build command; everything after --build-command is executed verbatim. "
            "The command must leave exactly one .whl in dist/ unless --input-wheel is used."
        ),
    )
    parser.add_argument("--output", help="final wheel filename; written under dist/ and defaults to the repaired manylinux wheel name")
    args = parser.parse_args()
    if args.input_wheel:
        args.skip_build = True
    if args.build_command == []:
        parser.error("--build-command requires a command to execute")
    return args


def main() -> int:
    args = parse_args()
    try:
        ensure_tool("auditwheel")
        ensure_tool("patchelf")
        clean_previous_outputs()
        input_wheel = build_wheel(args)

        env = auditwheel_env()
        work_dir = DIST_DIR / "publishable-work"
        work_dir.mkdir(parents=True, exist_ok=True)

        first_repair = auditwheel_repair(input_wheel, [], env)
        excludes = discover_excluded_libs(first_repair, work_dir)
        final_repair = auditwheel_repair(input_wheel, excludes, env)
        final_dir = copy_publishable_payload(final_repair, work_dir)

        output_wheel = final_output_path(final_repair, args.output)
        repack_wheel(final_dir, output_wheel)
        verify_wheel(output_wheel)
        cleanup_intermediate_outputs(output_wheel)
        log("done")
        return 0
    except (BuildError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
