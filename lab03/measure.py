from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown
import math
import json
import re
import os
import glob
import traceback
from pathlib import Path

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    iterations = []
    for i in range(repeats):
        start_time = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end_time = bench.clock()
        
        iterations.append((end_time - start_time)/1000000.0)
        
    return iterations


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("there are too few samples")

    # get second half of the samples + median
    second_half = samples[len(samples) // 2 :]
    median = statistics.median(second_half)
    
    if median <= 0:
        return unknown("median is less than or equal to zero")
    
    threshold = median * (1 + WARMUP_TOL)
    # find consecutive samples that are greater the threshold
    i = 0
    while samples[i] > threshold:
        i += 1
    
    result = {
        "settled_rate_ms": round(median, 4),
        "threshold_ms": round(threshold, 4),
        "tolerance": WARMUP_TOL,
        "retained": len(samples)-1
    }
    return measured(
        1,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        **result
    )

def summarize(samples: list[float]) -> dict[str, Any]:
    if samples is None or len(samples) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
        
    sortedarr = sorted(samples)
    mean = statistics.fmean(sortedarr)
    min_val = sortedarr[0]
    max_val = sortedarr[-1]
    standard_dev = statistics.stdev(sortedarr) if len(sortedarr) > 2 else 0.0
    
    # compute 50, 95, 99 percentiles using linear interpolation
    h50 = 0.5 * (len(sortedarr) - 1)
    h95 = 0.95*(len(sortedarr) - 1)
    h99 = 0.99*(len(sortedarr) - 1)
    
    i50 = math.floor(h50)
    i95 = math.floor(h95)
    i99 = math.floor(h99)

    p50 = round(sortedarr[i50] + (h50 - i50) * (sortedarr[i50 + 1] - sortedarr[i50]), 4)
    p95 = round(sortedarr[i95] + (h95 - i95) * (sortedarr[i95 + 1] - sortedarr[i95]), 4)
    p99 = round(sortedarr[i99] + (h99 - i99) * (sortedarr[i99 + 1] - sortedarr[i99]), 4)
    
    return {
        "n": len(sortedarr),
        "mean": round(mean, 4),
        "std": round(standard_dev, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "p50": p50,
        "p95": p95,
        "p99": p99
    }
        


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 20:
        return unknown("not enough samples for multimodal")
    
    sortedarr = sorted(samples)
    # trim lowest 5, highest 5 percent
    n = len(sortedarr)
    lower = int(n * 0.05)
    upper = int(n * 0.95)
    trimmed = sortedarr[lower:upper]
    
    # calculate gaps
    gaps = []
    for i in range(len(trimmed) - 1):
        gap = trimmed[i + 1] - trimmed[i]
        gaps.append(gap)
    
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return unknown("median gap is less than zero, timer resolution is too coarse")
    
    widest_gap = max(gaps)
    ratio = widest_gap / median_gap
    
    # determine split point where gap is made
    for i in range(len(trimmed) - 1):
        gap = trimmed[i + 1] - trimmed[i]
        if gap == widest_gap:
            left_of_gap = i
            right_of_gap = len(trimmed) - i    
    modes = []
    # left gap stats (0-i)
    upper = len(sortedarr) - upper
    modes.append({"n": left_of_gap+lower+1, "share": (left_of_gap+lower+1)/100, "median_ms": round(statistics.median(sortedarr[0:left_of_gap+lower+1]), 4)})
    # right gap stats (i-end)
    modes.append({"n": right_of_gap+upper-1, "share": (right_of_gap+upper-1)/100, "median_ms": round(statistics.median(sortedarr[left_of_gap+lower+1:]), 4)})

    result = {
        "gap_ratio": round(ratio, 2),
        "widest_gap_ms": round(widest_gap,3),
        "typical_gap_ms": round(median_gap,5),
        "modes": modes
    }
    return measured(
        True if ratio >= 20.0 and left_of_gap + lower >= 0.1 * len(sortedarr) and right_of_gap + upper >= 0.1 * len(sortedarr) else False,
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        **result)
        

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:

    findings = {}
    res = bench.runner(["nvpmodel", "-q"])
    if res.returncode != 0:
        return unknown("nvpmodel -q", res.error)
    
    data = res.stdout
    mode_re = re.compile(r"NV Power Mode:\s*([\d.]+)W\n\s*([\d.]+)")
    result = mode_re.search(data)
    if not result:
        return unknown("nvpmodel -q", "Power mode unable to be read from file")
    power_mode = result.group(1)
    findings["value"] = str(power_mode) + "W"

    findings["source"] = "nvpmodel -q"
    findings["status"] = "ok"

    mode_index = result.group(2)
    findings["mode_index"] = int(mode_index)

    root = Path("/")
    cpufreq_min = read_text(root, CPUFREQ_MIN)
    cpufreq_max = read_text(root, CPUFREQ_MAX)

    if not cpufreq_max or not cpufreq_min:
        findings["jetson_clocks"] = None
    elif cpufreq_max == cpufreq_min:
        findings["jetson_clocks"] = True
    else:
        findings["jetson_clocks"] = False

    findings["jetson_clocks_source"] = {
        "value": "scaling_min_freq=" + cpufreq_min + ", scaling_max_freq=" + cpufreq_max,
        "source": CPUFREQ_MIN + " vs " + CPUFREQ_MAX,
        "status": "ok"
    }

    return findings



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    result: dict[str, Any] = {}

    thermal_root = bench.telemetry / THERMAL_ZONES.lstrip("/")
    temps: list[float] = []
    max_temp = None
    zone_name = None

    if thermal_root.exists():
        for zone_dir in thermal_root.iterdir():
            temp_path = zone_dir / "temp"
            if not temp_path.is_file():
                continue
            try:
                temp_text = temp_path.read_text(errors="replace")
            except (OSError, TypeError, ValueError, UnicodeDecodeError):
                continue
            temp_text = temp_text.strip("\x00").strip()
            if not temp_text:
                continue
            try:
                temp_c = float(temp_text) / 1000.0
            except ValueError:
                continue
            if temp_c <= -1000.0:
                continue
            temps.append(temp_c)
            if max_temp is None or temp_c > max_temp:
                max_temp = temp_c
                zone_name = zone_dir.name

    if max_temp is None:
        result["temperature_c"] = unknown(
            "sys/devices/virtual/thermal/*/temp",
            "no readable thermal zones were found",
        )
    else:
        result["temperature_c"] = {
            "value": round(max_temp, 2),
            "source": "sys/devices/virtual/thermal/*/temp",
            "status": "ok",
            "zone": zone_name,
            "zones_read": len(temps),
        }

    power_source = " | ".join(POWER_RAIL_CANDIDATES)
    power_entry = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_entry is None:
        result["power_mw"] = unknown(
            power_source,
            "none of the documented INA3221 rail paths could be read",
        )
    else:
        _, power_text = power_entry
        try:
            power_mw = int(power_text)
        except ValueError:
            result["power_mw"] = unknown(
                power_source,
                "power rail value is not an integer in milliwatts",
            )
        else:
            result["power_mw"] = measured(power_mw, power_source)

    gpu_source = " | ".join(GPU_LOAD_CANDIDATES)
    gpu_entry = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_entry is None:
        result["gpu_utilization_percent"] = unknown(
            gpu_source,
            "none of the documented GPU load paths could be read",
        )
    else:
        _, gpu_text = gpu_entry
        try:
            gpu_util = float(gpu_text) / 10.0
        except ValueError:
            result["gpu_utilization_percent"] = unknown(
                gpu_entry[0],
                "GPU load value is not parseable as a number",
            )
        else:
            result["gpu_utilization_percent"] = measured(
                round(gpu_util, 1),
                gpu_entry[0],
                units="per-mille / 10",
            )

    return result

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)