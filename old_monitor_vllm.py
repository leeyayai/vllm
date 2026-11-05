#!/usr/bin/env python3
"""
vLLM + NVIDIA GPU monitor.
One run creates its own directory containing:
  - monitor.csv
  - summary.txt
  - gpu_utilization.png
  - gpu_memory.png
  - kv_cache_usage.png
  - waiting_requests.png
  - running_requests.png

Example:
python3 monitor_vllm.py \
  --concurrency 8 --input-len 4096 --output-len 1024 \
  --num-prompts 32 --request-rate inf --interval 1
Stop with Ctrl+C to write the summary and plots.
"""

import argparse
import csv
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_name(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9.-]+", "-", value)
    return value.strip("-") or "unknown"


parser = argparse.ArgumentParser(
    description="Monitor GPU and vLLM metrics; save CSV, summary and plots."
)
parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
parser.add_argument("--interval", type=float, default=1.0)
parser.add_argument("--output-root", default="./monitor_results")
parser.add_argument("--concurrency", type=int, required=True,
                    help="Benchmark --max-concurrency")
parser.add_argument("--input-len", type=int, required=True,
                    help="Benchmark --random-input-len")
parser.add_argument("--output-len", type=int, default=1024,
                    help="Benchmark --random-output-len")
parser.add_argument("--num-prompts", type=int, default=32)
parser.add_argument("--request-rate", default="inf")
parser.add_argument("--burstiness", default="1")
parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
parser.add_argument("--tag", default="", help="Optional extra label")
args = parser.parse_args()

if args.interval <= 0:
    parser.error("--interval must be greater than zero")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
parts = [
    f"concurrency{args.concurrency}",
    f"seq{args.input_len}",
    f"in{args.input_len}",
    f"out{args.output_len}",
    f"prompts{args.num_prompts}",
    f"rate{safe_name(args.request_rate)}",
    f"burst{safe_name(args.burstiness)}",
    f"model{safe_name(args.model)}",
]
if args.tag:
    parts.append(f"tag{safe_name(args.tag)}")
run_name = "monitor_" + "_".join(parts) + "_" + stamp
run_dir = Path(args.output_root) / run_name
run_dir.mkdir(parents=True, exist_ok=False)

csv_path = run_dir / "monitor.csv"
summary_path = run_dir / "summary.txt"

running = True
def stop_handler(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


def get_gpu():
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,"
        "power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip().splitlines()
        if not output:
            return [None] * 5
        # This script records the first GPU. For multi-GPU setups, use
        # --query-gpu=index,... and extend the CSV schema.
        values = [x.strip() for x in output[0].split(",")]
        return [float(x) for x in values]
    except Exception:
        return [None] * 5


def get_metrics():
    try:
        req = Request(args.url, headers={"User-Agent": "vllm-monitor"})
        with urlopen(req, timeout=3) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_prometheus(content):
    """Return metric_name -> list of values; labels are intentionally ignored."""
    result = {}
    pattern = re.compile(
        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+'
        r'([-+]?(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)'
        r'(?:\s+.*)?$'
    )
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        name, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        result.setdefault(name, []).append(value)
    return result


def get_metric(data, names, aggregate="sum"):
    for name in names:
        values = data.get(name)
        if values:
            if aggregate == "max":
                return max(values)
            if aggregate == "first":
                return values[0]
            return sum(values)
    return None


def get_metric_by_fragment(data, fragments, aggregate="sum"):
    matches = []
    for name, values in data.items():
        if any(fragment in name.lower() for fragment in fragments):
            matches.extend(values)
    if not matches:
        return None
    if aggregate == "max":
        return max(matches)
    return sum(matches)


def normalize_kv(value):
    # vLLM gpu_cache_usage_perc is normally a ratio in [0, 1].
    # Some exporters/versions may already expose percentage [0, 100].
    if value is not None and 0 <= value <= 1:
        return value * 100.0
    return value


def fmt(value, suffix=""):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}{suffix}"


fields = [
    "elapsed_seconds", "timestamp",
    "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
    "gpu_power_w", "gpu_temp_c",
    "kv_cache_pct", "running_requests", "waiting_requests",
    "preemptions_total", "preemptions_delta",
    "prompt_tokens_total", "prompt_tokens_delta",
    "generation_tokens_total", "generation_tokens_delta",
    "metrics_available",
]

samples = []
previous_counters = {}
start_time = time.monotonic()

print(f"\n监控目录：{run_dir}")
print(f"采样间隔：{args.interval}s")
print("开始采集；Benchmark 完成后按 Ctrl+C，自动生成 summary 和 5 张图。\n")

with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    try:
        while running:
            cycle_start = time.monotonic()
            elapsed = cycle_start - start_time
            timestamp = datetime.now().isoformat(timespec="seconds")

            gpu = get_gpu()
            metrics_text = get_metrics()
            metrics = parse_prometheus(metrics_text)

            kv = get_metric(metrics, [
                "vllm:gpu_cache_usage_perc",
                "vllm:kv_cache_usage_perc",
                "vllm:gpu_cache_usage_percent",
            ], aggregate="max")
            if kv is None:
                kv = get_metric_by_fragment(
                    metrics, ["gpu_cache_usage_perc", "kv_cache_usage_perc"],
                    aggregate="max"
                )
            kv = normalize_kv(kv)

            running_req = get_metric(metrics, [
                "vllm:num_requests_running",
            ])
            waiting_req = get_metric(metrics, [
                "vllm:num_requests_waiting",
            ])
            preemptions = get_metric(metrics, [
                "vllm:num_preemptions_total",
                "vllm:preemptions_total",
            ])
            prompt_tokens = get_metric(metrics, [
                "vllm:prompt_tokens_total",
                "vllm:prompt_tokens",
            ])
            generation_tokens = get_metric(metrics, [
                "vllm:generation_tokens_total",
                "vllm:generation_tokens",
            ])

            row = {
                "elapsed_seconds": round(elapsed, 3),
                "timestamp": timestamp,
                "gpu_util_pct": gpu[0],
                "gpu_mem_used_mb": gpu[1],
                "gpu_mem_total_mb": gpu[2],
                "gpu_mem_pct": (
                    gpu[1] / gpu[2] * 100
                    if gpu[1] is not None and gpu[2] else None
                ),
                "gpu_power_w": gpu[3],
                "gpu_temp_c": gpu[4],
                "kv_cache_pct": kv,
                "running_requests": running_req,
                "waiting_requests": waiting_req,
                "preemptions_total": preemptions,
                "preemptions_delta": None,
                "prompt_tokens_total": prompt_tokens,
                "prompt_tokens_delta": None,
                "generation_tokens_total": generation_tokens,
                "generation_tokens_delta": None,
                "metrics_available": bool(metrics),
            }

            for key, value in [
                ("preemptions", preemptions),
                ("prompt_tokens", prompt_tokens),
                ("generation_tokens", generation_tokens),
            ]:
                if value is not None and key in previous_counters:
                    row[key + "_delta"] = value - previous_counters[key]
                if value is not None:
                    previous_counters[key] = value

            writer.writerow(row)
            csv_file.flush()
            samples.append(row)

            print(
                f"\r{timestamp[11:]} | "
                f"GPU {fmt(gpu[0], '%'):>7} | "
                f"显存 {fmt(gpu[1], 'MB'):>10}/{fmt(gpu[2], 'MB')} | "
                f"功耗 {fmt(gpu[3], 'W'):>8} | "
                f"温度 {fmt(gpu[4], 'C'):>7} | "
                f"KV {fmt(kv, '%'):>7} | "
                f"Running {fmt(running_req):>6} | "
                f"Waiting {fmt(waiting_req):>6}",
                end="", flush=True
            )

            duration = time.monotonic() - cycle_start
            time.sleep(max(0, args.interval - duration))
    except KeyboardInterrupt:
        running = False

print("\n\n采集结束，正在生成汇总和图片……")


def valid_values(key):
    values = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            continue
        try:
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        except (TypeError, ValueError):
            pass
    return values


def stats_text(key, suffix=""):
    values = valid_values(key)
    if not values:
        return "无有效数据"
    return (
        f"min={min(values):.2f}{suffix}, "
        f"avg={sum(values)/len(values):.2f}{suffix}, "
        f"max={max(values):.2f}{suffix}"
    )


def counter_change(key):
    values = valid_values(key)
    if len(values) < 2:
        return "无有效数据"
    return f"{values[-1] - values[0]:.0f}"


metadata = [
    f"实验目录名: {run_name}",
    f"Model: {args.model}",
    f"Concurrency: {args.concurrency}",
    f"Input length: {args.input_len}",
    f"Output length: {args.output_len}",
    f"Num prompts: {args.num_prompts}",
    f"Request rate: {args.request_rate}",
    f"Burstiness: {args.burstiness}",
    f"Sampling interval: {args.interval}s",
    f"Samples: {len(samples)}",
    "",
    "[GPU]",
    f"GPU Utilization (%): {stats_text('gpu_util_pct', '%')}",
    f"GPU Memory Used (MB): {stats_text('gpu_mem_used_mb', ' MB')}",
    f"GPU Memory Used (%): {stats_text('gpu_mem_pct', '%')}",
    f"Power (W): {stats_text('gpu_power_w', ' W')}",
    f"Temperature (C): {stats_text('gpu_temp_c', ' C')}",
    "",
    "[vLLM]",
    f"KV Cache Usage (%): {stats_text('kv_cache_pct', '%')}",
    f"Running Requests: {stats_text('running_requests')}",
    f"Waiting Requests: {stats_text('waiting_requests')}",
    f"Preemptions delta during monitoring: {counter_change('preemptions_total')}",
    f"Prompt Tokens delta during monitoring: {counter_change('prompt_tokens_total')}",
    f"Generation Tokens delta during monitoring: {counter_change('generation_tokens_total')}",
    "",
    "注意：Counter 增量为本次监控首末值之差；如果监控期间有其他请求，",
    "这些 Token/Preemption 也会被计入。建议每次只运行一组 Benchmark。",
    "缺失的 vLLM 指标会显示为 N/A；请检查当前版本 /metrics 的实际名称。",
]
summary_path.write_text("\n".join(metadata), encoding="utf-8")
print("\n".join(metadata))


def make_plot(key, title, ylabel, filename):
    x = valid_values("elapsed_seconds")
    y = valid_values(key)
    # Keep x/y paired, rather than independently filtering missing values.
    pairs = []
    for sample in samples:
        xv, yv = sample.get("elapsed_seconds"), sample.get(key)
        if xv is None or yv is None:
            continue
        try:
            xv, yv = float(xv), float(yv)
            if math.isfinite(xv) and math.isfinite(yv):
                pairs.append((xv, yv))
        except (TypeError, ValueError):
            continue

    if not pairs:
        print(f"跳过图片 {filename}：没有有效数据")
        return

    px, py = zip(*pairs)
    plt.figure(figsize=(11, 4.8))
    plt.plot(px, py, linewidth=1.4)
    plt.title(title)
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / filename, dpi=150)
    plt.close()


make_plot("gpu_util_pct", "GPU Utilization over Time",
          "GPU Utilization (%)", "gpu_utilization.png")
make_plot("gpu_mem_used_mb", "GPU Memory Usage over Time",
          "GPU Memory Used (MB)", "gpu_memory.png")
make_plot("kv_cache_pct", "vLLM KV Cache Usage over Time",
          "KV Cache Usage (%)", "kv_cache_usage.png")
make_plot("waiting_requests", "vLLM Waiting Requests over Time",
          "Waiting Requests", "waiting_requests.png")
make_plot("running_requests", "vLLM Running Requests over Time",
          "Running Requests", "running_requests.png")

print(f"\n完成！所有结果保存在：{run_dir.resolve()}")
print(f"  - {csv_path.name}")
print(f"  - {summary_path.name}")
print("  - gpu_utilization.png")
print("  - gpu_memory.png")
print("  - kv_cache_usage.png")
print("  - waiting_requests.png")
print("  - running_requests.png")
