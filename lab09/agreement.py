
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
import statistics
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    ms = list(ms)
    for x in ms:
        x = _finite(x, "latency report")
        if x < 0:
            raise AgreementError("non-monotonic clock")

    if not ms:
        raise AgreementError("sequence is empty")

    ms = sorted(ms)

    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    p99 = _percentile(ms, 99)

    return {
        "n": len(ms),
        "mean": statistics.mean(ms),
        "worst": max(ms),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": p95 - p50
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base_ms = _finite(base_ms, "speedup base_ms")
    other_ms = _finite(other_ms, "speedup other_ms")

    if base_ms <= 0 or other_ms <= 0:
        raise AgreementError("clock resolution error or skipped execution")

    factor = base_ms / other_ms

    faster = True if factor > 1.0 else False

    saved_ms = base_ms - other_ms

    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved_ms
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError("batching inconsistencies or missing tensor flattening")
    if len(a) == 0:
        raise AgreementError("sequences must be non-empty")

    for x, y in zip(a,b):
        x = _finite(x, "max_abs_dev x")
        y = _finite(y, "max_abs_dev y")

    best_i = 0
    best = abs(a[0] - b[0])

    for i in range(1, len(a)):
        if abs(a[i] - b[i]) > best:
            best_i = i
            best = abs(a[i]-b[i])

    return {
        "max_abs": best,
        "index": best_i,
        "a": a[best_i],
        "b": b[best_i],
        "n": len(a),
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    floor = _finite(floor, "max_rel_dev floor")
    if floor <= 0:
        raise AgreementError("relative-error floor must be positive")
    if len(a) == 0 or len(b) == 0:
        raise AgreementError("sequences must be non-empty")
    if len(a) != len(b):
        raise AgreementError(
            "batching inconsistencies or missing tensor flattening"
        )

    values_a = []
    for i, x in enumerate(a):
        values_a.append(_finite(x, f"max_rel_dev a[{i}]"))

    values_b = []
    for i, x in enumerate(b):
        values_b.append(_finite(x, f"max_rel_dev b[{i}]"))

    best = -1.0
    best_i = 0
    floored = 0
    denom_min = float("inf")
    for i, (x, y) in enumerate(zip(values_a, values_b)):
        denominator = abs(y)
        denom_min = min(denom_min, denominator)
        if denominator < floor:
            floored += 1
        relative = abs(x - y) / max(denominator, floor)
        if relative > best:
            best = relative
            best_i = i

    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": len(values_a),
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(rows_a) == 0 or len(rows_b) == 0:
        raise AgreementError("row sequences must be non-empty")
    if len(rows_a) != len(rows_b):
        raise AgreementError("row counts do not match")

    agree = 0
    first_disagreement = None
    for k, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        if len(row_a) == 0 or len(row_b) == 0:
            raise AgreementError(f"row {k} is empty")
        if len(row_a) != len(row_b):
            raise AgreementError(f"class counts do not match in row {k}")

        scores_a = [
            _finite(value, f"rows_a[{k}][{i}]")
            for i, value in enumerate(row_a)
        ]
        scores_b = [
            _finite(value, f"rows_b[{k}][{i}]")
            for i, value in enumerate(row_b)
        ]
        predicted_a = max(range(len(scores_a)), key=scores_a.__getitem__)
        predicted_b = max(range(len(scores_b)), key=scores_b.__getitem__)
        if predicted_a == predicted_b:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    n = len(rows_a)
    return {
        "n": n,
        "agree": agree,
        "disagree": n - agree,
        "rate": agree / n,
        "first_disagreement": first_disagreement,
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    tol = _finite(tol, "decision_margin tol")
    if tol < 0:
        raise AgreementError("deviation tolerance must be non-negative")
    if len(rows) == 0:
        raise AgreementError("row sequence must be non-empty")

    margins = []
    for k, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError(f"row {k} must contain at least two scores")
        scores = [
            _finite(value, f"rows[{k}][{i}]")
            for i, value in enumerate(row)
        ]
        top_two = sorted(scores, reverse=True)[:2]
        margins.append(top_two[0] - top_two[1])

    min_margin = min(margins)
    argmin = margins.index(min_margin)
    below_tol = sum(margin < tol for margin in margins)
    return {
        "n": len(margins),
        "min": min_margin,
        "median": _percentile(margins, 50),
        "mean": float(sum(margins) / len(margins)),
        "argmin": argmin,
        "below_tol": below_tol,
        "safe": below_tol == 0,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()