#!/usr/bin/env python3
import argparse
import json
import os
import time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--limit-gib", type=float, default=4.0)
    p.add_argument("--start-epoch-file", default="")
    return p.parse_args()

def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def read_start_epoch(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return None

def print_peak_total_memory_line(peak_system_used_kb: int, mem_total_kb: int):
    if peak_system_used_kb <= 0:
        return
    if mem_total_kb > 0:
        used_pct = peak_system_used_kb * 100.0 / mem_total_kb
        print(f"[sgl-mem] Peak total system memory usage: {peak_system_used_kb / 1024.0 / 1024.0:.2f} GiB ({used_pct:.1f}%)")
    else:
        print(f"[sgl-mem] Peak total system memory usage: {peak_system_used_kb / 1024.0 / 1024.0:.2f} GiB")

def main():
    args = parse_args()
    limit_kb = int(args.limit_gib * 1024 * 1024)
    start_epoch = read_start_epoch(args.start_epoch_file)
    sep = "=" * 72

    print(sep)
    print("[sgl-mem] Build Memory Peak Report")
    print(sep)
    print(f"[sgl-mem] File peak RSS threshold: {args.limit_gib:.1f} GiB")

    if not os.path.exists(args.db):
        print("[sgl-mem] No compile memory records found.")
        if start_epoch is not None:
            print(f"[sgl-mem] Total build time: {format_duration(time.time() - start_epoch)}")
        print(sep)
        return 0

    peaks = {}
    peak_system_used_kb = 0
    mem_total_kb = 0
    with open(args.db, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = str(rec.get("source", "<unknown>")).replace("\\u0000", "")
            peak = int(rec.get("peak_rss_kb", 0))
            if peak > peaks.get(src, 0):
                peaks[src] = peak
            rec_used = int(rec.get("peak_system_used_kb", 0))
            rec_total = int(rec.get("mem_total_kb", 0))
            if rec_used > peak_system_used_kb:
                peak_system_used_kb = rec_used
            if rec_total > mem_total_kb:
                mem_total_kb = rec_total

    offenders = [(src, kb) for src, kb in peaks.items() if kb >= limit_kb]
    offenders.sort(key=lambda x: x[1], reverse=True)

    if not offenders:
        print("[sgl-mem] No files exceeded the threshold.")
        if start_epoch is not None:
            print(f"[sgl-mem] Total build time: {format_duration(time.time() - start_epoch)}")
        print_peak_total_memory_line(peak_system_used_kb, mem_total_kb)
        print(sep)
        return 0

    print("[sgl-mem] Files above threshold (peak RSS):")
    for src, kb in offenders:
        print(f"[sgl-mem]   {src}: {kb / 1024.0 / 1024.0:.2f} GiB")

    if start_epoch is not None:
        print(f"[sgl-mem] Total build time: {format_duration(time.time() - start_epoch)}")
    print_peak_total_memory_line(peak_system_used_kb, mem_total_kb)
    print(sep)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
