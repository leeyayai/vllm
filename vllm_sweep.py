#!/usr/bin/env python3
"""
vLLM Serving Benchmark Sweep
─────────────────────────────
Runs `vllm bench serve` over a Cartesian product of parameter values,
streams each run's output live to the terminal, and collates every run's
metrics into a timestamped CSV + JSON in --output-dir.

Optional integrated monitoring
  Add --monitor to launch monitor_vllm.py alongside every benchmark run.
  The monitor is started before the benchmark, stopped cleanly afterwards,
  and its output directory is recorded in the CSV / JSON.

Prerequisites:
  • A vLLM API server must already be running (e.g. `vllm serve <model>`).
  • `vllm` must be on PATH.

Usage:
    python vllm_sweep.py                    # bare sweep
    python vllm_sweep.py --dry-run          # print commands, don't run
    python vllm_sweep.py --monitor          # sweep + per-run monitoring
    python vllm_sweep.py --monitor \\
        --monitor-script ./monitor_vllm.py \\
        --monitor-interval 1 \\
        --monitor-warmup 3
"""

import argparse
import csv
import itertools
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# 1.  PARAMETER GRID  —  edit these lists to define your sweep.
#     Every combination in the Cartesian product will be run.
# ══════════════════════════════════════════════════════════════
PARAM_GRID: dict[str, list] = {
    "random_input_len":  [16384],           # e.g. [4096, 8192, 16384]
    "random_output_len": [1024],            # e.g. [512, 1024, 2048]
    "num_prompts":       [32],              # e.g. [16, 32, 64]
    "request_rate":      ["inf"],           # e.g. ["inf", "8", "4"]
    "max_concurrency":   [1],               # e.g. [1, 4, 8]
}

# ══════════════════════════════════════════════════════════════
# 2.  FIXED SETTINGS  —  shared across every run
# ══════════════════════════════════════════════════════════════
FIXED = {
    "model":              "/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    "host":               "127.0.0.1",
    "port":               "8000",
    "dataset_name":       "random",
    "metric_percentiles": "50,90,99",
    "result_dir":         "./bench_results",
}

# Short model label derived from the path, used as the monitor's --model arg.
_path_parts = Path(FIXED["model"]).parts
MONITOR_MODEL_LABEL: str = next(
    (p for p in reversed(_path_parts) if p not in {"master", "main", "snapshots"}),
    "model",
)

# ══════════════════════════════════════════════════════════════
# 3.  METRIC PATTERNS  —  regex → column name
# ══════════════════════════════════════════════════════════════
METRIC_RE: dict[str, str] = {
    # Overview
    "successful_requests":   r"Successful requests:\s+([\d.]+)",
    "duration_s":            r"Benchmark duration \(s\):\s+([\d.]+)",
    "total_input_tokens":    r"Total input tokens:\s+([\d.]+)",
    "total_output_tokens":   r"Total generated tokens:\s+([\d.]+)",
    "req_throughput":        r"Request throughput \(req/s\):\s+([\d.]+)",
    "output_tok_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "total_tok_throughput":  r"Total Token throughput \(tok/s\):\s+([\d.]+)",
    # Time to First Token
    "ttft_mean_ms":          r"Mean TTFT \(ms\):\s+([\d.]+)",
    "ttft_p50_ms":           r"P50 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p90_ms":           r"P90 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p99_ms":           r"P99 TTFT \(ms\):\s+([\d.]+)",
    # Time per Output Token
    "tpot_mean_ms":          r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_p50_ms":           r"P50 TPOT \(ms\):\s+([\d.]+)",
    "tpot_p90_ms":           r"P90 TPOT \(ms\):\s+([\d.]+)",
    "tpot_p99_ms":           r"P99 TPOT \(ms\):\s+([\d.]+)",
    # Inter-token Latency
    "itl_mean_ms":           r"Mean ITL \(ms\):\s+([\d.]+)",
    "itl_p50_ms":            r"P50 ITL \(ms\):\s+([\d.]+)",
    "itl_p90_ms":            r"P90 ITL \(ms\):\s+([\d.]+)",
    "itl_p99_ms":            r"P99 ITL \(ms\):\s+([\d.]+)",
}

# Columns shown in the terminal summary table (keeps it readable)
SUMMARY_DISPLAY_COLS = [
    "req_throughput",
    "output_tok_throughput",
    "ttft_mean_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p99_ms",
    "itl_mean_ms",
    "itl_p99_ms",
]


# ──────────────────────────────────────────────────────────────
# benchmark helpers
# ──────────────────────────────────────────────────────────────

def build_cmd(params: dict) -> list[str]:
    """Assemble the vllm bench serve command for a parameter dict."""
    return [
        "vllm", "bench", "serve",
        "--model",              FIXED["model"],
        "--host",               FIXED["host"],
        "--port",               FIXED["port"],
        "--dataset-name",       FIXED["dataset_name"],
        "--random-input-len",   str(params["random_input_len"]),
        "--random-output-len",  str(params["random_output_len"]),
        "--num-prompts",        str(params["num_prompts"]),
        "--request-rate",       str(params["request_rate"]),
        "--max-concurrency",    str(params["max_concurrency"]),
        "--metric-percentiles", FIXED["metric_percentiles"],
        "--save-result",
        "--result-dir",         FIXED["result_dir"],
    ]


def parse_metrics(text: str) -> dict:
    """Extract all benchmark metrics from stdout text."""
    out: dict[str, float | None] = {}
    for name, pattern in METRIC_RE.items():
        m = re.search(pattern, text)
        out[name] = float(m.group(1)) if m else None
    return out


# ──────────────────────────────────────────────────────────────
# monitor helpers
# ──────────────────────────────────────────────────────────────

def start_monitor(
    params: dict,
    monitor_script: str,
    run_monitor_root: Path,
    interval: float,
) -> subprocess.Popen:
    """
    Launch monitor_vllm.py as a background subprocess.

    The monitor's --output-root is set to `run_monitor_root`, a directory
    unique to this sweep run.  The monitor creates one timestamped sub-
    directory inside it.  After the monitor exits, callers can find that
    directory by listing run_monitor_root.
    """
    run_monitor_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, monitor_script,
        "--concurrency",   str(params["max_concurrency"]),
        "--input-len",     str(params["random_input_len"]),
        "--output-len",    str(params["random_output_len"]),
        "--num-prompts",   str(params["num_prompts"]),
        "--request-rate",  str(params["request_rate"]),
        "--output-root",   str(run_monitor_root),
        "--interval",      str(interval),
        "--model",         MONITOR_MODEL_LABEL,
    ]

    # Redirect monitor's verbose status output to a sidecar log file so it
    # doesn't interleave with the benchmark's streaming output.
    log_path = run_monitor_root / "monitor_stdout.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        text=True,
    )
    # Attach the file handle so stop_monitor can close it.
    proc._sweep_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def stop_monitor(proc: subprocess.Popen, timeout: float = 90.0) -> None:
    """
    Send SIGINT to the monitor subprocess and wait for it to finish
    writing its CSV, summary and dashboard PNG.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Monitor did not stop within {timeout}s — killing it.")
        proc.kill()
        proc.wait()
    except Exception as exc:
        print(f"  ⚠️  Error stopping monitor: {exc}")
    finally:
        log_fh = getattr(proc, "_sweep_log_fh", None)
        if log_fh and not log_fh.closed:
            log_fh.close()


def find_monitor_run_dir(run_monitor_root: Path) -> str:
    """Return the path of the single subdirectory the monitor created."""
    if not run_monitor_root.exists():
        return str(run_monitor_root)
    subdirs = sorted(
        (d for d in run_monitor_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return str(subdirs[-1]) if subdirs else str(run_monitor_root)


# ──────────────────────────────────────────────────────────────
# core run logic
# ──────────────────────────────────────────────────────────────

def run_one(
    params: dict,
    idx: int,
    total: int,
    *,
    output_dir: str = ".",
    monitor_script: str | None = None,
    monitor_interval: float = 1.0,
    monitor_warmup: float = 2.0,
) -> dict:
    """
    Run a single benchmark combination.

    - Streams stdout live to the terminal and captures it for metric parsing.
    - Optionally starts monitor_vllm.py before the benchmark and stops it
      cleanly after, recording the monitor's output directory in the result.
    - Returns a flat dict of params + metrics (+ monitor_dir if applicable).
    """
    banner = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{'━' * 72}", flush=True)
    print(f"  Run {idx}/{total}  │  {banner}", flush=True)
    print(f"{'━' * 72}", flush=True)

    cmd = build_cmd(params)
    print("  CMD:", " ".join(cmd), "\n", flush=True)

    # ── optional: start monitor ───────────────────────────────────────────────
    monitor_proc: subprocess.Popen | None = None
    run_monitor_root: Path | None = None

    if monitor_script:
        if not Path(monitor_script).is_file():
            print(f"  ⚠️  Monitor script not found: {monitor_script} — skipping monitor.")
        else:
            run_monitor_root = Path(output_dir) / "monitor" / f"run_{idx:03d}"
            print(
                f"  🔍 Starting monitor (warmup {monitor_warmup}s) …\n"
                f"     script : {monitor_script}\n"
                f"     root   : {run_monitor_root}",
                flush=True,
            )
            monitor_proc = start_monitor(
                params, monitor_script, run_monitor_root, monitor_interval
            )
            time.sleep(monitor_warmup)
            if monitor_proc.poll() is not None:
                print(
                    f"  ⚠️  Monitor exited early (rc={monitor_proc.returncode}). "
                    f"Check {run_monitor_root}/monitor_stdout.log"
                )
                monitor_proc = None
            else:
                print(f"  🔍 Monitor PID {monitor_proc.pid} running.\n", flush=True)

    # ── run benchmark ─────────────────────────────────────────────────────────
    lines: list[str] = []
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so progress bars appear
            text=True,
            bufsize=1,                  # line-buffered for real-time streaming
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        proc.wait()
        returncode = proc.returncode
    except FileNotFoundError:
        msg = "ERROR: 'vllm' not found — is it installed and on PATH?"
        print(msg)
        lines.append(msg)

    # ── stop monitor ──────────────────────────────────────────────────────────
    monitor_dir: str | None = None

    if monitor_proc is not None:
        print(f"\n  🔍 Sending SIGINT to monitor (PID {monitor_proc.pid}) …", flush=True)
        stop_monitor(monitor_proc)
        monitor_dir = find_monitor_run_dir(run_monitor_root)
        print(f"  🔍 Monitor output: {monitor_dir}", flush=True)

    # ── parse & return ────────────────────────────────────────────────────────
    output  = "".join(lines)
    metrics = parse_metrics(output)
    status  = "ok" if returncode == 0 else f"error({returncode})"

    return {
        "run":          idx,
        **params,
        "status":       status,
        "returncode":   returncode,
        "monitor_dir":  monitor_dir,
        **metrics,
    }


# ──────────────────────────────────────────────────────────────
# output / reporting
# ──────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    """Format a value for the summary table."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def save_results(rows: list[dict], output_dir: str, timestamp: str) -> None:
    """Write full results to CSV and JSON (overwrites on each incremental call)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    csv_path  = out / f"sweep_{timestamp}.csv"
    json_path = out / f"sweep_{timestamp}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\n  💾  Saved → {csv_path}")
    print(f"  💾  Saved → {json_path}", flush=True)


def print_summary_table(rows: list[dict], show_monitor: bool = False) -> None:
    """Print a compact summary table to stdout after the sweep completes."""
    if not rows:
        return

    param_keys = list(PARAM_GRID.keys())
    all_cols   = param_keys + SUMMARY_DISPLAY_COLS

    # Compute column widths (max of header width vs widest data value)
    widths = {
        col: max(len(col), max(len(_fmt(r.get(col))) for r in rows))
        for col in all_cols
    }

    sep = "  ".join("-" * widths[c] for c in all_cols)
    hdr = "  ".join(c.ljust(widths[c]) for c in all_cols)

    print(f"\n{'═' * (len(sep) + 3)}")
    print("  SWEEP SUMMARY")
    print(f"{'═' * (len(sep) + 3)}")
    print(f"   {hdr}")
    print(f"   {sep}")

    for r in rows:
        ok   = "✓" if r.get("returncode") == 0 else "✗"
        line = "  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in all_cols)
        mdir = f"  → {r['monitor_dir']}" if show_monitor and r.get("monitor_dir") else ""
        print(f" {ok} {line}{mdir}")

    print(f"{'═' * (len(sep) + 3)}\n")

    print("  Metric key:")
    for col, desc in {
        "req_throughput":        "Request throughput (req/s)",
        "output_tok_throughput": "Output token throughput (tok/s)",
        "ttft_mean_ms":          "Mean time-to-first-token (ms)",
        "ttft_p99_ms":           "P99  time-to-first-token (ms)",
        "tpot_mean_ms":          "Mean time-per-output-token (ms)",
        "tpot_p99_ms":           "P99  time-per-output-token (ms)",
        "itl_mean_ms":           "Mean inter-token latency (ms)",
        "itl_p99_ms":            "P99  inter-token latency (ms)",
    }.items():
        print(f"    {col:<26} {desc}")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run vllm bench serve across a parameter grid and collate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── sweep options ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run without actually running them.",
    )
    ap.add_argument(
        "--output-dir", default=FIXED["result_dir"],
        help="Directory to write CSV / JSON summaries (default: %(default)s).",
    )
    # ── monitor options ───────────────────────────────────────────────────────
    ap.add_argument(
        "--monitor", action="store_true",
        help="Launch monitor_vllm.py alongside every benchmark run.",
    )
    ap.add_argument(
        "--monitor-script", default="./monitor_vllm.py",
        help="Path to monitor_vllm.py (default: %(default)s).",
    )
    ap.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="Monitor sampling interval in seconds (default: %(default)s).",
    )
    ap.add_argument(
        "--monitor-warmup", type=float, default=2.0,
        help=(
            "Seconds to wait after starting the monitor before the benchmark "
            "begins, so the monitor can capture the idle baseline (default: %(default)s)."
        ),
    )
    args = ap.parse_args()

    # Build Cartesian product of all parameter combinations
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*PARAM_GRID.values())]
    total  = len(combos)

    print(f"\n🚀  vLLM Benchmark Sweep  —  {total} combination(s) to run")
    print(f"    Model  : {FIXED['model']}")
    print(f"    Server : {FIXED['host']}:{FIXED['port']}")
    print("    Grid   :")
    for k, v in PARAM_GRID.items():
        print(f"      {k:<22} {v}")
    if args.monitor:
        print(f"    Monitor: {args.monitor_script}")
        print(f"      interval={args.monitor_interval}s  warmup={args.monitor_warmup}s")

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] — commands that would be executed:\n")
        for i, params in enumerate(combos, 1):
            print(f"  [{i:>{len(str(total))}}/{total}]  " + " ".join(build_cmd(params)))
        if args.monitor:
            print(
                "\n  [monitor] monitor_vllm.py would be started / stopped around each run."
            )
        return

    # ── live sweep ────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for i, params in enumerate(combos, 1):
        try:
            row = run_one(
                params, i, total,
                output_dir=args.output_dir,
                monitor_script=args.monitor_script if args.monitor else None,
                monitor_interval=args.monitor_interval,
                monitor_warmup=args.monitor_warmup,
            )
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted — saving partial results …")
            break
        except Exception as exc:
            print(f"❌  Run {i} failed with exception: {exc}")
            row = {
                "run":         i,
                **params,
                "status":      f"exception: {exc}",
                "returncode":  -1,
                "monitor_dir": None,
                **{k: None for k in METRIC_RE},
            }

        rows.append(row)
        # Save after every run so a crash never loses completed data
        save_results(rows, args.output_dir, timestamp)

    # ── final summary ─────────────────────────────────────────────────────────
    print_summary_table(rows, show_monitor=args.monitor)

    succeeded = sum(1 for r in rows if r.get("returncode") == 0)
    print(f"🏁  Sweep complete — {succeeded}/{total} runs succeeded.")
    print(f"    Results in: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()
