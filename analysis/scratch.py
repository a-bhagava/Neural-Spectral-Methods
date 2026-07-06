"""
Aggregates mixed-scale-training eval results into a single combined JSON:

    {model_type}/{train_combo}/{test_scale} -> {in_dist, n_seeds, <metric>_mean, <metric>_std, ...}

Expected directory layout (one level per component):

    <ROOT>/<model_type>/train_<combo>/test_<test_scale>/<run_dir>/metrics.json

e.g.
    eval_results_mixed_scale_training/nsm/train_combined_8_2/test_L=1.0/nsm_hdim32_..._22/metrics.json

Each <run_dir> is treated as one seed. All run_dirs under the same
(model_type, combo, test_scale) are averaged together (mean/std) to produce
the final entry.

ASSUMPTION ABOUT metrics.json CONTENTS: each individual metrics.json is
assumed to hold flat scalar values, e.g.:

    {"erra": 0.314, "errr": 0.520, "residual": 0.271}

If your files already look like {"erra_mean": ..., "erra_std": ...} (e.g.
because a run itself averaged multiple internal samples), the script also
handles that case by using those values directly as one "seed" datapoint
per run_dir (see `extract_raw_metrics` below) -- adjust that function if
your schema differs.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_TYPES = ["nsm", "unet"]
ROOT = Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/eval_results_mixed_scale_training")
OUT_PATH = ROOT / "mixed_scale_summary.json"


def parse_combo_scales(combo_name: str) -> list:
    """
    'combined_8_4_2' -> [0.8, 0.4, 0.2]
    'combined_8_2'   -> [0.8, 0.2]
    """
    parts = combo_name.split("_")
    # first part is literally "combined" -> skip it
    digits = parts[1:]
    return [int(d) / 10 for d in digits]


def parse_test_scale(test_scale_name: str) -> float:
    """
    'L=1.0' -> 1.0
    """
    match = re.search(r"L=([\d.]+)", test_scale_name)
    if not match:
        raise ValueError(f"Could not parse test scale from '{test_scale_name}'")
    return float(match.group(1))


def is_in_dist(combo_name: str, test_scale_name: str, tol: float = 1e-6) -> bool:
    train_scales = parse_combo_scales(combo_name)
    test_scale = parse_test_scale(test_scale_name)
    return any(abs(test_scale - ts) < tol for ts in train_scales)


def extract_raw_metrics(data: dict) -> dict:
    """
    Pull out {metric_name: value} from a single run's metrics.json.
    Handles both flat scalars ({"erra": ...}) and pre-averaged
    ({"erra_mean": ...}) schemas -- extend this if your files differ.
    """
    out = {}
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            continue
        if key.endswith("_mean"):
            out[key[: -len("_mean")]] = value
        elif key.endswith("_std"):
            continue  # skip, we recompute std across seeds ourselves
        else:
            out[key] = value
    return out


def aggregate(root: Path, model_types: list) -> dict:
    agg = {}

    for model_type in model_types:
        model_dir = root / model_type
        if not model_dir.is_dir():
            print(f"[warn] missing model dir: {model_dir}")
            continue

        agg[model_type] = {}

        for combo_dir in sorted(model_dir.glob("train_*")):
            combo_name = combo_dir.name[len("train_"):]
            agg[model_type][combo_name] = {}

            for test_dir in sorted(combo_dir.glob("test_*")):
                test_scale_name = test_dir.name[len("test_"):]

                # collect one raw-metric dict per seed/run_dir
                seed_metrics = defaultdict(list)  # metric_name -> [values across seeds]
                run_dirs = [d for d in test_dir.iterdir() if d.is_dir()]

                for run_dir in run_dirs:
                    metrics_path = run_dir / "metrics.json"
                    if not metrics_path.exists():
                        print(f"[warn] no metrics.json in {run_dir}")
                        continue
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                    raw = extract_raw_metrics(data)
                    if not raw:
                        print(f"[warn] no numeric metrics found in {metrics_path} "
                              f"(keys present: {list(data.keys())})")
                        continue
                    for metric_name, value in raw.items():
                        seed_metrics[metric_name].append(value)

                if not seed_metrics:
                    print(f"[warn] no usable runs under {test_dir}")
                    continue

                n_seeds = max(len(v) for v in seed_metrics.values())
                entry = {
                    "in_dist": is_in_dist(combo_name, test_scale_name),
                    "n_seeds": n_seeds,
                }
                for metric_name, values in seed_metrics.items():
                    arr = np.array(values, dtype=float)
                    entry[f"{metric_name}_mean"] = float(arr.mean())
                    entry[f"{metric_name}_std"] = float(arr.std())

                agg[model_type][combo_name][test_scale_name] = entry

    return agg


if __name__ == "__main__":
    result = aggregate(ROOT, MODEL_TYPES)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote combined summary -> {OUT_PATH}")