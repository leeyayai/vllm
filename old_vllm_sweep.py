#!/usr/bin/env python3
"""
vLLM Serving Benchmark Sweep
─────────────────────────────
Runs `vllm bench serve` over a Cartesian product of parameter values,
streams each run's output live to the terminal, and collates every run's
metrics into a timestamped CSV + JSON in --output-dir.

Prerequisites:
  • A vLLM API server must already be running (e.g. `vllm serve <model>`).
  • `vllm` must be on PATH.

Usage:
    python vllm_sweep.py              # full sweep
    python vllm_sweep.py --dry-run   # just print the commands, don't run
    python vllm_sweep.py --output-dir ./my_results
"""

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# 1.  PARAMETER GRID  —  edit these lists to define your sweep.
#     Every combination in the Cartesian product will be run.
# ══════════════════════════════════════════════════════════════
PARAM_GRID: dict[str, list] = {
    "random_input_len":  [4096,16384],           # e.g. [4096, 8192, 16384]
    "random_output_len": [1024],            # e.g. [512, 1024, 2048]
    "num_prompts":       [32],              # e.g. [16, 32, 64]
    "request_rate":      ["inf"],           # e.g. ["inf", "8", "4"]
    "max_concurrency":   [1,8],               # e.g. [1, 4, 8]
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
    "ttft_median_ms":        r"Median TTFT \(ms\):\s+([\d.]+)",
    "ttft_p50_ms":           r"P50 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p90_ms":           r"P90 TTFT \(ms\):\s+([\d.]+)",
    "ttft_p99_ms":           r"P99 TTFT \(ms\):\s+([\d.]+)",
    # Time per Output Token
    "tpot_mean_ms":          r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_median_ms":        r"Median TPOT \(ms\):\s+([\d.]+)",
    "tpot_p50_ms":           r"P50 TPOT \(ms\):\s+([\d.]+)",
    "tpot_p90_ms":           r"P90 TPOT \(ms\):\s+([\d.]+)",
    "tpot_p99_ms":           r"P99 TPOT \(ms\):\s+([\d.]+)",
    # Inter-token Latency
    "itl_mean_ms":           r"Mean ITL \(ms\):\s+([\d.]+)",
    "itl_median_ms":         r"Median ITL \(ms\):\s+([\d.]+)",
    "itl_p50_ms":            r"P50 ITL \(ms\):\s+([\d.]+)",
    "itl_p90_ms":            r"P90 ITL \(ms\):\s+([\d.]+)",
    "itl_p99_ms":            r"P99 ITL \(ms\):\s+([\d.]+)",
}

# Columns shown in the terminal summary table (keep it readable)
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
# helpers
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


def run_one(params: dict, idx: int, total: int) -> dict:
    """
    Run a single benchmark combination.
    Streams stdout live to the terminal and captures it for parsing.
    Returns a flat dict of params + metrics.
    """
    banner = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{'━' * 70}", flush=True)
    print(f"  Run {idx}/{total}  │  {banner}", flush=True)
    print(f"{'━' * 70}", flush=True)

    cmd = build_cmd(params)
    print("  CMD:", " ".join(cmd), "\n", flush=True)

    lines: list[str] = []
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered
        )
        # Stream line-by-line so the user sees progress in real time
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

    output   = "".join(lines)
    metrics  = parse_metrics(output)
    status   = "ok" if returncode == 0 else f"error({returncode})"
    return {"run": idx, **params, "status": status, "returncode": returncode, **metrics}


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


def print_summary_table(rows: list[dict]) -> None:
    """Print a compact summary table to stdout."""
    if not rows:
        return

    param_keys = list(PARAM_GRID.keys())
    all_cols   = param_keys + SUMMARY_DISPLAY_COLS

    # Compute column widths (header vs. data)
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
        print(f" {ok} {line}")

    print(f"{'═' * (len(sep) + 3)}")
    print()

    # Column key
    print("  Column key:")
    col_descriptions = {
        "req_throughput":        "Request throughput (req/s)",
        "output_tok_throughput": "Output token throughput (tok/s)",
        "ttft_mean_ms":          "Mean time-to-first-token (ms)",
        "ttft_p99_ms":           "P99 time-to-first-token (ms)",
        "tpot_mean_ms":          "Mean time-per-output-token (ms)",
        "tpot_p99_ms":           "P99 time-per-output-token (ms)",
        "itl_mean_ms":           "Mean inter-token latency (ms)",
        "itl_p99_ms":            "P99 inter-token latency (ms)",
    }
    for col, desc in col_descriptions.items():
        print(f"    {col:<26} {desc}")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run vllm bench serve across a parameter grid and collate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run without actually running them.",
    )
    ap.add_argument(
        "--output-dir", default=FIXED["result_dir"],
        help="Directory to write CSV / JSON summaries (default: %(default)s).",
    )
    args = ap.parse_args()

    # Build Cartesian product of all parameter combinations
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*PARAM_GRID.values())]
    total  = len(combos)

    print(f"\n🚀  vLLM Benchmark Sweep  —  {total} combination(s) to run")
    print(f"    Model : {FIXED['model']}")
    print(f"    Server: {FIXED['host']}:{FIXED['port']}")
    print("    Grid  :")
    for k, v in PARAM_GRID.items():
        print(f"      {k:<22} {v}")

    # ── dry run ──────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] — commands that would be executed:\n")
        for i, params in enumerate(combos, 1):
            print(f"  [{i:>{len(str(total))}}/{total}]  " + " ".join(build_cmd(params)))
        return

    # ── live run ─────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for i, params in enumerate(combos, 1):
        try:
            row = run_one(params, i, total)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted — saving partial results …")
            break
        except Exception as exc:
            print(f"❌  Run {i} failed with exception: {exc}")
            row = {
                "run": i,
                **params,
                "status": f"exception: {exc}",
                "returncode": -1,
                **{k: None for k in METRIC_RE},
            }

        rows.append(row)
        # Save after every run so a crash never loses data
        save_results(rows, args.output_dir, timestamp)

    # ── final summary ────────────────────────────────────────
    print_summary_table(rows)

    succeeded = sum(1 for r in rows if r.get("returncode") == 0)
    print(f"🏁  Sweep complete — {succeeded}/{total} runs succeeded.")
    print(f"    Results in: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()
