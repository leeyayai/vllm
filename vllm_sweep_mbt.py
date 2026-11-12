#!/usr/bin/env python3
"""
vLLM --max-num-batched-tokens Sweep
───────────────────────────────────
For every value of --max-num-batched-tokens this script:

  1. starts   `vllm serve <model> --max-num-batched-tokens <N>`  (own process group)
  2. waits    until the server's /health endpoint returns 200
  3. runs     the full parameter grid defined in vllm_sweep.py
              (optionally with monitor_vllm.py around every run)
  4. stops    the server, waits for GPU memory + port to be released
  5. moves on to the next value

vllm_sweep.py is imported unchanged, so PARAM_GRID / FIXED (model, host, port ...),
the metric regexes, the CSV/JSON columns and the monitor output are exactly the
same as before -- plus one extra column: max_num_batched_tokens.

Put this file next to vllm_sweep.py and monitor_vllm.py.

Usage:
    python vllm_sweep_mbt.py \\
        --max-num-batched-tokens 512 1024 2048 4096 8192 \\
        --monitor \\
        --monitor-script ./monitor_vllm.py \\
        --monitor-interval 1 \\
        --monitor-warmup 3 \\
        --output-dir ./bench_results

    python vllm_sweep_mbt.py --dry-run --max-num-batched-tokens 512 1024

Output layout:
    <output-dir>/
        sweep_mbt_<timestamp>.csv / .json     <- all batches in one table
        mbt_512/
            vllm_server.log                   <- server stdout/stderr
            <vllm bench result *.json>
            monitor/run_001/monitor_.../      <- monitor.csv, summary.txt, dashboard.png
        mbt_1024/
            ...
"""

import argparse
import itertools
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Make `import vllm_sweep` work no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vllm_sweep as vs  # noqa: E402


# ══════════════════════════════════════════════════════════════
# Default values to sweep (override with --max-num-batched-tokens)
# ══════════════════════════════════════════════════════════════
DEFAULT_MBT_VALUES = [512, 1024, 2048, 4096, 8192]

# After the server is stopped, GPU memory must fall back to within this many MB
# of the pre-launch level before the next server is started.
GPU_RELEASE_TOLERANCE_MB = 500


# ──────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────

def port_in_use(host: str, port: str | int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, int(port))) == 0


def gpu_mem_used_mb() -> float | None:
    """Total used GPU memory (MB) summed over all GPUs, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().splitlines()
        return sum(float(x) for x in out)
    except Exception:
        return None


def tail_file(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join("     │ " + ln for ln in lines[-n:])


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ──────────────────────────────────────────────────────────────
# server lifecycle
# ──────────────────────────────────────────────────────────────

def build_serve_cmd(mbt: int, extra_args: list[str]) -> list[str]:
    return [
        "vllm", "serve", vs.FIXED["model"],
        "--host", vs.FIXED["host"],
        "--port", str(vs.FIXED["port"]),
        "--max-num-batched-tokens", str(mbt),
        *extra_args,
    ]


def start_server(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start vllm serve in its own session/process group (so we can kill all workers)."""
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._server_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def wait_ready(proc: subprocess.Popen, log_path: Path, timeout: float) -> bool:
    """Poll /health until 200, the process dies, or the timeout expires."""
    url = f"http://{vs.FIXED['host']}:{vs.FIXED['port']}/health"
    t0 = time.monotonic()
    last_note = t0
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            print(f"  ❌ Server exited early (rc={proc.returncode}). Last log lines:")
            print(tail_file(log_path), flush=True)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"  ✅ Server ready after {time.monotonic() - t0:.0f}s", flush=True)
                    return True
        except Exception:
            pass
        if time.monotonic() - last_note >= 15:
            print(f"     … waiting for server ({time.monotonic() - t0:.0f}s)", flush=True)
            last_note = time.monotonic()
        time.sleep(2)
    print(f"  ❌ Server not ready within {timeout:.0f}s. Last log lines:")
    print(tail_file(log_path), flush=True)
    return False


def stop_server(
    proc: subprocess.Popen | None,
    baseline_mem_mb: float | None,
    timeout: float = 90.0,
) -> None:
    """SIGINT the server (same as Ctrl+C), escalate to SIGKILL, then wait for resources."""
    if proc is None:
        return
    pgid = proc.pid  # start_new_session=True -> pgid == pid
    print(f"\n  🛑 Stopping vllm server (PID {proc.pid}) …", flush=True)
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"  ⚠️  Server did not stop within {timeout:.0f}s — SIGKILL.")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

        # Give leftover workers a moment to exit on their own, then kill the rest.
        deadline = time.monotonic() + 15
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.5)
        if _group_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        fh = getattr(proc, "_server_log_fh", None)
        if fh and not fh.closed:
            fh.close()

    # Wait for the port and the GPU memory to be released.
    t0 = time.monotonic()
    while port_in_use(vs.FIXED["host"], vs.FIXED["port"]) and time.monotonic() - t0 < 30:
        time.sleep(1)

    if baseline_mem_mb is None:
        time.sleep(5)
    else:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            used = gpu_mem_used_mb()
            if used is not None and used <= baseline_mem_mb + GPU_RELEASE_TOLERANCE_MB:
                break
            time.sleep(1)
        else:
            print(
                f"  ⚠️  GPU memory still {gpu_mem_used_mb()} MB "
                f"(baseline {baseline_mem_mb:.0f} MB) — continuing anyway."
            )
    print("  🛑 Server stopped.", flush=True)


# ──────────────────────────────────────────────────────────────
# row helpers
# ──────────────────────────────────────────────────────────────

def with_mbt(row: dict, mbt: int) -> dict:
    """Insert the max_num_batched_tokens column right after `run`."""
    out = {"run": row["run"], "max_num_batched_tokens": mbt}
    out.update({k: v for k, v in row.items() if k != "run"})
    return out


def failed_row(idx: int, mbt: int, params: dict, status: str) -> dict:
    return with_mbt(
        {
            "run": idx,
            **params,
            "status": status,
            "returncode": -1,
            "monitor_dir": None,
            **{k: None for k in vs.METRIC_RE},
        },
        mbt,
    )


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep `vllm serve --max-num-batched-tokens`, running the vllm_sweep.py "
                    "grid (+ optional monitor) against a freshly started server for each value.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── server sweep options ──────────────────────────────────────────────────
    ap.add_argument(
        "--max-num-batched-tokens", type=int, nargs="+", default=DEFAULT_MBT_VALUES,
        metavar="N",
        help="Values to sweep, e.g. 512 1024 2048 (default: %(default)s).",
    )
    ap.add_argument(
        "--extra-serve-args", default="",
        help='Extra args appended to `vllm serve`, e.g. '
             '"--max-model-len 20000 --gpu-memory-utilization 0.9".',
    )
    ap.add_argument(
        "--server-ready-timeout", type=float, default=900.0,
        help="Max seconds to wait for each server to become ready (default: %(default)s).",
    )
    ap.add_argument(
        "--cooldown", type=float, default=5.0,
        help="Seconds to sleep between batches (default: %(default)s).",
    )
    # ── same options as vllm_sweep.py ─────────────────────────────────────────
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would be run without running them.")
    ap.add_argument("--output-dir", default=vs.FIXED["result_dir"],
                    help="Directory for CSV / JSON summaries and per-batch folders "
                         "(default: %(default)s).")
    ap.add_argument("--monitor", action="store_true",
                    help="Launch monitor_vllm.py alongside every benchmark run.")
    ap.add_argument("--monitor-script", default="./monitor_vllm.py",
                    help="Path to monitor_vllm.py (default: %(default)s).")
    ap.add_argument("--monitor-interval", type=float, default=1.0,
                    help="Monitor sampling interval in seconds (default: %(default)s).")
    ap.add_argument("--monitor-warmup", type=float, default=2.0,
                    help="Seconds to wait after starting the monitor before the benchmark "
                         "begins (default: %(default)s).")
    args = ap.parse_args()

    mbt_values = list(dict.fromkeys(args.max_num_batched_tokens))   # de-dup, keep order
    extra_args = shlex.split(args.extra_serve_args)
    output_dir = Path(args.output_dir)

    # Parameter grid comes straight from vllm_sweep.py
    keys   = list(vs.PARAM_GRID.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*vs.PARAM_GRID.values())]
    total  = len(mbt_values) * len(combos)

    print(f"\n🚀  vLLM max-num-batched-tokens Sweep  —  "
          f"{len(mbt_values)} server config(s) × {len(combos)} combo(s) = {total} run(s)")
    print(f"    Model  : {vs.FIXED['model']}")
    print(f"    Server : {vs.FIXED['host']}:{vs.FIXED['port']}")
    print(f"    MBT    : {mbt_values}")
    if extra_args:
        print(f"    Extra serve args: {extra_args}")
    print("    Grid   :")
    for k, v in vs.PARAM_GRID.items():
        print(f"      {k:<22} {v}")
    if args.monitor:
        print(f"    Monitor: {args.monitor_script}")
        print(f"      interval={args.monitor_interval}s  warmup={args.monitor_warmup}s")

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] — commands that would be executed:\n")
        n = 0
        for mbt in mbt_values:
            print(f"  ── max-num-batched-tokens = {mbt}")
            print("     [serve] " + " ".join(build_serve_cmd(mbt, extra_args)))
            for params in combos:
                n += 1
                print(f"     [{n:>{len(str(total))}}/{total}] " + " ".join(vs.build_cmd(params)))
            print("     [stop ] SIGINT vllm serve, wait for GPU memory to be released\n")
        if args.monitor:
            print("  [monitor] monitor_vllm.py would be started / stopped around each run.")
        return

    # ── pre-flight checks ─────────────────────────────────────────────────────
    if shutil.which("vllm") is None:
        sys.exit("ERROR: 'vllm' not found — is it installed and on PATH?")
    if port_in_use(vs.FIXED["host"], vs.FIXED["port"]):
        sys.exit(
            f"ERROR: {vs.FIXED['host']}:{vs.FIXED['port']} is already in use. "
            "Stop the running `vllm serve` first — otherwise the benchmark would hit "
            "the wrong server."
        )

    # ── live sweep ────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_label = vs.MONITOR_MODEL_LABEL
    rows: list[dict] = []
    global_idx = 0
    interrupted = False

    try:
        for b, mbt in enumerate(mbt_values, 1):
            print(f"\n{'█' * 72}", flush=True)
            print(f"  Batch {b}/{len(mbt_values)}  │  max-num-batched-tokens = {mbt}", flush=True)
            print(f"{'█' * 72}", flush=True)

            batch_dir = output_dir / f"mbt_{mbt}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            log_path = batch_dir / "vllm_server.log"

            # Per-batch settings injected into vllm_sweep's module globals:
            #   * vllm bench JSON results go into this batch's folder
            #   * the monitor's --model label carries the mbt value, so it shows up
            #     in the monitor directory name and on the dashboard
            vs.FIXED["result_dir"] = str(batch_dir)
            vs.MONITOR_MODEL_LABEL = f"{base_label}-mbt{mbt}"

            serve_cmd = build_serve_cmd(mbt, extra_args)
            print("  SERVE:", " ".join(serve_cmd))
            print(f"  LOG  : {log_path}\n", flush=True)

            baseline_mem = gpu_mem_used_mb()
            server = None
            try:
                server = start_server(serve_cmd, log_path)
                ready = wait_ready(server, log_path, args.server_ready_timeout)

                if not ready:
                    for params in combos:
                        global_idx += 1
                        rows.append(failed_row(global_idx, mbt, params, "server_start_failed"))
                    vs.save_results(rows, args.output_dir, f"mbt_{timestamp}")
                else:
                    for params in combos:
                        global_idx += 1
                        try:
                            row = vs.run_one(
                                params, global_idx, total,
                                output_dir=str(batch_dir),
                                monitor_script=args.monitor_script if args.monitor else None,
                                monitor_interval=args.monitor_interval,
                                monitor_warmup=args.monitor_warmup,
                            )
                            row = with_mbt(row, mbt)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            print(f"❌  Run {global_idx} failed with exception: {exc}")
                            row = failed_row(global_idx, mbt, params, f"exception: {exc}")
                        rows.append(row)
                        # Save after every run so a crash never loses completed data
                        vs.save_results(rows, args.output_dir, f"mbt_{timestamp}")
            finally:
                stop_server(server, baseline_mem)

            if b < len(mbt_values):
                time.sleep(args.cooldown)

    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠️  Interrupted — saving partial results …")
        if rows:
            vs.save_results(rows, args.output_dir, f"mbt_{timestamp}")

    # ── final summary ─────────────────────────────────────────────────────────
    # print_summary_table() takes its parameter columns from vs.PARAM_GRID, so
    # prepend max_num_batched_tokens there (the combos were already generated).
    vs.PARAM_GRID = {"max_num_batched_tokens": mbt_values, **vs.PARAM_GRID}
    vs.print_summary_table(rows, show_monitor=args.monitor)

    succeeded = sum(1 for r in rows if r.get("returncode") == 0)
    tag = " (interrupted)" if interrupted else ""
    print(f"🏁  Sweep complete{tag} — {succeeded}/{total} runs succeeded.")
    print(f"    Results in: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
