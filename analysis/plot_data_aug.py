# Compare the best NSM data-augmentation run (trained on L=0.8, evaluated on
# all test scales) against the L=0.8 single-scale baselines for both NSM and
# UNet -- one subplot per metric, 3 bars per test-scale group.

import json
import numpy as _np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Global font styling (matches previous scripts) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"

TRAIN_SCALE = "L=0.8"  # the data-aug run was trained on this scale

AUG_COLOR          = "#16a34a"  # green -- highlights the augmented run
BASELINE_NSM_COLOR = "#8e44ad"  # purple -- matches NSM family from earlier plots
BASELINE_UNET_COLOR = "#ea580c"  # orange -- matches UNet family from earlier plots


def load_agg(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def discover_metrics(*aggs_or_dicts) -> list:
    """
    Discover metric names (keys ending in '_mean'). Accepts either full
    agg dicts (agg[model][train][test] = entry) or a flat dict of
    test_scale -> entry (like the data-aug aggregation below).
    """
    metrics = set()

    def scan_entry(entry):
        for key in entry:
            if key.endswith("_mean"):
                metrics.add(key[: -len("_mean")])

    for obj in aggs_or_dicts:
        # flat {test_scale: entry}
        if obj and all(isinstance(v, dict) and "n_seeds" in v for v in obj.values()):
            for entry in obj.values():
                scan_entry(entry)
        else:
            # nested agg[model][train][test] = entry
            for model_type in obj.values():
                for train_key in model_type.values():
                    for entry in train_key.values():
                        scan_entry(entry)
    return sorted(metrics)


def extract_raw_metrics(data: dict) -> dict:
    """Flat scalar keys ({'erra': ..., 'errr': ...}) -> {metric_name: value}."""
    out = {}
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            continue
        if key.endswith("_mean"):
            out[key[: -len("_mean")]] = value
        elif key.endswith("_std"):
            continue
        else:
            out[key] = value
    return out


def aggregate_data_aug(root: Path, train_scale: str = TRAIN_SCALE) -> dict:
    """
    Walk <root>/test_L=X[/<run_dir>]/metrics.json and build:
        {test_scale: {in_dist, n_seeds, <metric>_mean, <metric>_std, ...}}

    Handles both layouts:
      root/test_L=0.1/metrics.json                (single run, no seed folder)
      root/test_L=0.1/<run_dir>/metrics.json       (one or more seeds)
    """
    result = {}

    for test_dir in sorted(root.glob("test_*")):
        test_scale = test_dir.name[len("test_"):]

        seed_values = []  # list of {metric_name: value} dicts, one per seed

        direct_metrics = test_dir / "metrics.json"
        if direct_metrics.exists():
            with open(direct_metrics, "r") as f:
                seed_values.append(extract_raw_metrics(json.load(f)))
        else:
            for run_dir in test_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                with open(metrics_path, "r") as f:
                    seed_values.append(extract_raw_metrics(json.load(f)))

        if not seed_values:
            print(f"[warn] no metrics.json found under {test_dir}")
            continue

        # collect per-metric values across seeds
        by_metric = {}
        for sv in seed_values:
            for metric_name, value in sv.items():
                by_metric.setdefault(metric_name, []).append(value)

        entry = {
            "in_dist": (test_scale == train_scale),
            "n_seeds": len(seed_values),
        }
        for metric_name, values in by_metric.items():
            arr = _np.array(values, dtype=float)
            entry[f"{metric_name}_mean"] = float(arr.mean())
            entry[f"{metric_name}_std"] = float(arr.std())

        result[test_scale] = entry

    return result


def _grouped_bar_subplot(ax, series, metric, test_scales):
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = _np.arange(len(test_scales))

    for i, s in enumerate(series):
        means, stds, in_dists = [], [], []
        for test_scale in test_scales:
            entry = s["data"].get(test_scale)
            if entry is None or f"{metric}_mean" not in entry:
                means.append(_np.nan); stds.append(0.0); in_dists.append(False)
            else:
                means.append(entry[f"{metric}_mean"])
                stds.append(entry.get(f"{metric}_std", 0.0))
                in_dists.append(entry.get("in_dist", False))

        offset = (i - (n_series - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=s["label"], color=s["color"],
            edgecolor="none", linewidth=0,
        )
        for xi, mean, std, in_dist in zip(x + offset, means, stds, in_dists):
            if in_dist and not _np.isnan(mean):
                ax.annotate("*", (xi, mean + std), ha="center", va="bottom",
                            fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(test_scales, fontweight="bold")
    ax.set_xlabel("Test scale", fontweight="bold")


def plot_data_augmented_eval(
    aug_data: dict,
    single_agg: dict,
    output_dir: Path,
    train_scale: str = TRAIN_SCALE,
    metrics: list = None,
    test_scales: list = None,
    title: str = None,
    ncols: int = None,
):
    """
    Side-by-side comparison: augmented NSM (trained on `train_scale`) vs the
    non-augmented NSM and UNet baselines trained on the same scale. One
    subplot per metric, 3 bars per test-scale group.

    aug_data:   {test_scale: entry}                          from aggregate_data_aug()
    single_agg: agg[model_type][train_scale][test_scale]      from cross_scale_summary.json
    """
    metrics = metrics or discover_metrics(aug_data, single_agg)

    nsm_baseline = single_agg.get("nsm", {}).get(train_scale, {})
    unet_baseline = single_agg.get("unet", {}).get(train_scale, {})

    test_scales = test_scales or sorted(set().union(
        aug_data.keys(), nsm_baseline.keys(), unet_baseline.keys()
    ))

    title = title or f"Data augmentation vs baselines (train {train_scale})"

    series = [
        {"label": "NSM (augmented)", "color": AUG_COLOR, "data": aug_data},
        {"label": "NSM (baseline)", "color": BASELINE_NSM_COLOR, "data": nsm_baseline},
        {"label": "UNet (baseline)", "color": BASELINE_UNET_COLOR, "data": unet_baseline},
    ]

    n = len(metrics)
    ncols = ncols or n
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        _grouped_bar_subplot(ax, series, metric, test_scales)
        metric_label = metric.upper() if len(metric) <= 6 else metric.replace("_", " ").title()
        ax.set_title(metric_label, fontweight="bold")
        ax.set_ylabel(f"{metric_label} (lower = better)", fontweight="bold")

    for ax in axes_flat[n:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    star_handle = plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10)
    legend = fig.legend(
        handles + [star_handle], labels + ["in-distribution"],
        loc="upper center", ncol=len(labels) + 1, bbox_to_anchor=(0.5, 1.15),
        frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.suptitle(title, y=1.25, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(output_dir) / "data_augmented_vs_baselines.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {out_path}")
    return out_path


if __name__ == "__main__":
    aug_root = Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/eval_results_data_aug/best_run_cross_scale")
    aug_data = aggregate_data_aug(aug_root, train_scale=TRAIN_SCALE)

    single_agg = load_agg(Path("eval_results_cross/cross_scale_summary.json"))

    plot_data_augmented_eval(
        aug_data, single_agg,
        output_dir=Path("eval_results_data_aug"),
        train_scale=TRAIN_SCALE,
    )