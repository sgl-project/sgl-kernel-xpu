#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict

SRC_EXTS = (".cpp", ".cc", ".cxx", ".c", ".sycl", ".S", ".s")
HEADER = "Memory"


def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--db",
        default=os.path.join(os.getcwd(), "build", "mem_monitor", "compile_mem.jsonl"),
    )
    p.add_argument("--file-limit-gib", type=float, default=4.0)
    p.add_argument(
        "--file-limit-fatal",
        action="store_true",
        help="Kill the compile and fail the build if a single file's peak RSS "
        "reaches --file-limit-gib (instead of only reporting it).",
    )
    p.add_argument("--guard-avail-gib", type=float, default=0.5)
    p.add_argument("--guard-used-pct", type=float, default=99.0)
    args, rest = p.parse_known_args()
    if not rest:
        p.error("missing compiler command")
    args.compiler_cmd = rest
    return args


def read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, v = line.split(":", 1)
                out[k] = int(v.strip().split()[0])
    except Exception:
        return out
    return out


def ppid_of(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            s = f.read()
        r = s.rfind(")")
        rest = s[r + 2 :].split()
        return int(rest[1])
    except Exception:
        return None


def descendants(root_pid):
    children = defaultdict(list)
    try:
        proc_entries = os.listdir("/proc")
    except Exception:
        return set()
    for name in proc_entries:
        if not name.isdigit():
            continue
        pid = int(name)
        pp = ppid_of(pid)
        if pp is not None:
            children[pp].append(pid)
    out = set()
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def vmrss_kb(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


def tree_rss_kb(root_pid):
    total = 0
    for pid in descendants(root_pid):
        total += vmrss_kb(pid)
    return total


def detect_source(args):
    best = None
    for a in args:
        if a.endswith(SRC_EXTS):
            best = a
            if os.path.basename(a) != a:
                return os.path.basename(a)
    if best:
        return os.path.basename(best)
    for i, a in enumerate(args):
        if a in ("-o", "--output") and i + 1 < len(args):
            return os.path.basename(args[i + 1])
    return os.path.basename(args[0]) if args else "<unknown>"


def terminate(proc):
    """SIGTERM the compiler, then SIGKILL if it does not exit promptly."""
    try:
        proc.send_signal(signal.SIGTERM)
        time.sleep(1.0)
        if proc.poll() is None:
            proc.kill()
    except ProcessLookupError:
        pass


def append_record(db_path, record):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    lock_path = db_path + ".lock"
    with open(lock_path, "a", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        with open(db_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\\n")
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def print_current_offenders(db_path, file_limit_kb):
    peaks = {}
    if not os.path.exists(db_path):
        return
    try:
        with open(db_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                src = rec.get("source", "<unknown>")
                peak = int(rec.get("peak_rss_kb", 0))
                if peak > peaks.get(src, 0):
                    peaks[src] = peak
    except Exception:
        return
    offenders = [(s, p) for s, p in peaks.items() if p >= file_limit_kb]
    offenders.sort(key=lambda x: x[1], reverse=True)
    if not offenders:
        return
    print(f"[{HEADER}] Current files above limit:", file=sys.stderr)
    for src, peak in offenders[:10]:
        print(f"[{HEADER}]   {src}: {peak / 1024.0:.1f} MiB", file=sys.stderr)


def main():
    args = parse_args()
    source = detect_source(args.compiler_cmd[1:])
    file_limit_kb = int(args.file_limit_gib * 1024 * 1024)
    mem = read_meminfo()
    mem_total_kb = int(mem.get("MemTotal", 0))
    guard_avail_kb = int(args.guard_avail_gib * 1024 * 1024)

    proc = subprocess.Popen(args.compiler_cmd)
    peak_kb = 0
    peak_system_used_kb = 0
    killed_by_guard = False
    guard_reason = ""
    file_limit_exceeded = False

    while proc.poll() is None:
        rss_kb = tree_rss_kb(proc.pid)
        if rss_kb > peak_kb:
            peak_kb = rss_kb

        mem = read_meminfo()
        avail_kb = int(mem.get("MemAvailable", 0))
        used_kb = max(mem_total_kb - avail_kb, 0)
        if used_kb > peak_system_used_kb:
            peak_system_used_kb = used_kb
        used_pct = 0.0 if mem_total_kb == 0 else used_kb * 100.0 / mem_total_kb

        # Per-file hard limit: kill as soon as a single compile's tree RSS
        # reaches the threshold so the memory is actually reclaimed (not just
        # reported after the fact) and the build fails fast.
        if args.file_limit_fatal and rss_kb >= file_limit_kb:
            file_limit_exceeded = True
            print(
                f"[{HEADER}] {source} reached the per-file limit "
                f"({rss_kb / 1024.0 / 1024.0:.2f} GiB >= {args.file_limit_gib:.1f} GiB), "
                "stopping build. Split this translation unit into smaller TUs.",
                file=sys.stderr,
            )
            terminate(proc)
            break

        if avail_kb <= guard_avail_kb or used_pct >= args.guard_used_pct:
            killed_by_guard = True
            guard_reason = (
                f"avail={avail_kb / 1024.0 / 1024.0:.2f}GiB used={used_pct:.1f}%"
            )
            print(
                f"[{HEADER}] OOM guard hit while building {source} ({guard_reason}), stopping build.",
                file=sys.stderr,
            )
            terminate(proc)
            break

        time.sleep(0.2)

    ret = proc.wait()
    rec = {
        "source": source,
        "peak_rss_kb": peak_kb,
        "peak_system_used_kb": peak_system_used_kb,
        "mem_total_kb": mem_total_kb,
        "exit_code": ret,
        "guard_triggered": killed_by_guard,
        "guard_reason": guard_reason,
        "file_limit_exceeded": file_limit_exceeded,
        "cmd": " ".join(args.compiler_cmd),
    }
    append_record(args.db, rec)
    if killed_by_guard or file_limit_exceeded:
        print_current_offenders(args.db, file_limit_kb)
    # Signal failure to ninja/cmake on a limit breach even if the killed
    # compiler happened to exit 0 in the race before the signal landed.
    if file_limit_exceeded:
        return ret or 3
    return ret


if __name__ == "__main__":
    sys.exit(main())
