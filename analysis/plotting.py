# # generate bar charts for all the models we've trained

# # single scale - evaluated on all scales for both NSM and UNet
# # mixed scale training - evaluated on all scales compared to single scale training, separate by model - raw metrics
# # data augmented run for NSM - trained on L=0.8 and tested on all, compared to equivalent NSM model without augmentation

# import json
# import numpy as _np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # --- Global font styling ---
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.titleweight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["figure.titleweight"] = "bold"

# MODEL_TYPES = ["nsm", "unet"]

# # Distinct purples for NSM (light -> dark), distinct oranges for UNet (light -> dark),
# # one shade per train scale so 3 train-scale x 2 model combos are all distinguishable.
# MODEL_TRAIN_SCALE_COLORS = {
#     "nsm":  ["#c9a3e8", "#8e44ad", "#4a1d6e"],   # light -> dark purple
#     "unet": ["#fdba74", "#ea580c", "#7c2d12"],   # light -> dark orange
# }


# def load_agg(json_path: Path) -> dict:
#     with open(json_path, "r") as f:
#         return json.load(f)


# def discover_metrics(agg: dict) -> list:
#     metrics = set()
#     for model_type in agg.values():
#         for train_scale in model_type.values():
#             for entry in train_scale.values():
#                 for key in entry:
#                     if key.endswith("_mean"):
#                         metrics.add(key[: -len("_mean")])
#     return sorted(metrics)


# def _bar_subplot_multi_train(ax, agg, metric, train_scales, test_scales, model_types):
#     """
#     Draw grouped bars for ALL (train_scale x model_type) combos on one axis,
#     grouped by test_scale on the x-axis. e.g. 3 train scales x 2 models = 6
#     bars per test-scale group. Color encodes both model (hue family) and
#     train scale (shade within family). No outline on the bars.
#     """
#     n_groups = len(train_scales) * len(model_types)
#     bar_width = 0.8 / n_groups
#     x = _np.arange(len(test_scales))

#     combo_idx = 0
#     for ti, train_scale in enumerate(train_scales):
#         for model_type in model_types:
#             shades = MODEL_TRAIN_SCALE_COLORS.get(model_type, ["#888888"] * len(train_scales))
#             color = shades[ti % len(shades)]

#             means, stds, in_dists = [], [], []
#             for test_scale in test_scales:
#                 entry = agg.get(model_type, {}).get(train_scale, {}).get(test_scale)
#                 if entry is None or f"{metric}_mean" not in entry:
#                     means.append(_np.nan); stds.append(0.0); in_dists.append(False)
#                 else:
#                     means.append(entry[f"{metric}_mean"])
#                     stds.append(entry.get(f"{metric}_std", 0.0))
#                     in_dists.append(entry["in_dist"])

#             offset = (combo_idx - (n_groups - 1) / 2) * bar_width
#             ax.bar(
#                 x + offset, means, bar_width,
#                 yerr=stds, capsize=2,
#                 label=f"{model_type.upper()} (train {train_scale})",
#                 color=color,
#                 edgecolor="none",
#                 linewidth=0,
#             )

#             for xi, mean, std, in_dist in zip(x + offset, means, stds, in_dists):
#                 if in_dist and not _np.isnan(mean):
#                     ax.annotate("*", (xi, mean + std), ha="center", va="bottom",
#                                 fontsize=12, fontweight="bold")

#             combo_idx += 1

#     ax.set_xticks(x)
#     ax.set_xticklabels(test_scales, fontweight="bold")
#     ax.set_xlabel("Test scale", fontweight="bold")


# def plot_all_scales_eval(
#     agg: dict,
#     output_dir: Path,
#     metrics: list = None,
#     train_scales: list = None,
#     test_scales: list = None,
#     model_types: list = None,
#     title: str = None,
#     ncols: int = None,
# ):
#     """
#     One subplot per metric. Each subplot groups bars by test scale, with
#     (train_scale x model_type) combos side by side — e.g. 3 train scales x
#     2 models = 6 bars per test-scale group. Color families distinguish model
#     (purple = NSM, orange = UNet); shade within a family encodes train scale.
#     Error bars from std across seeds, star marks in-distribution.
#     """
#     model_types = model_types or MODEL_TYPES
#     metrics = metrics or discover_metrics(agg)
#     train_scales = train_scales or sorted(
#         set().union(*[agg.get(mt, {}).keys() for mt in model_types])
#     )
#     test_scales = test_scales or sorted(
#         set().union(*[
#             agg.get(mt, {}).get(ts, {}).keys()
#             for mt in model_types for ts in train_scales
#         ])
#     )
#     title = title or "Cross-scale eval: NSM vs UNet (all train scales)"

#     n = len(metrics)
#     ncols = ncols or n
#     nrows = -(-n // ncols)

#     fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
#     axes_flat = axes.flatten()

#     for ax, metric in zip(axes_flat, metrics):
#         _bar_subplot_multi_train(ax, agg, metric, train_scales, test_scales, model_types)
#         metric_label = metric.upper() if len(metric) <= 6 else metric.replace("_", " ").title()
#         ax.set_title(metric_label, fontweight="bold")
#         ax.set_ylabel(f"{metric_label} (lower = better)", fontweight="bold")

#     for ax in axes_flat[n:]:
#         ax.axis("off")

#     handles, labels = axes_flat[0].get_legend_handles_labels()
#     star_handle = plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10)
#     legend = fig.legend(
#         handles + [star_handle], labels + ["in-distribution"],
#         loc="upper center", ncol=min(len(labels) + 1, 4), bbox_to_anchor=(0.5, 1.15),
#         frameon=False, fontsize=9,
#     )
#     for text in legend.get_texts():
#         text.set_fontweight("bold")

#     fig.suptitle(title, y=1.25, fontsize=14, fontweight="bold")
#     fig.tight_layout()

#     out_path = Path(output_dir) / "all_scales_eval.png"
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     print(f"All-scales eval plot -> {out_path}")
#     return out_path


# def plot_mixed_scale_eval():
#     pass


# def plot_data_augmented_eval():
#     pass


# if __name__ == "__main__":
#     agg = load_agg(Path("eval_results_cross/cross_scale_summary.json"))
#     plot_all_scales_eval(agg, output_dir=Path("eval_results_cross"))

# generate bar charts for all the models we've trained
#
# 1. plot_all_scales_eval        - single-scale training, all train scales x both models, one subplot/metric
# 2. plot_mixed_scale_by_model   - ONE model at a time: all 6 training conditions
#                                   (3 single-scale + 3 mixed-scale combos), evaluated across
#                                   all test scales, one subplot per metric
# 3. plot_mixed_vs_models        - mixed-scale combos ONLY (no single-scale), both models
#                                   on the same plot, one subplot per metric

import json
import re
import numpy as _np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Global font styling ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"

MODEL_TYPES = ["nsm", "unet"]

# --- Palette 1: single-scale training, all train scales x both models (plot_all_scales_eval) ---
# purple family = NSM, orange family = UNet (unchanged from before)
MODEL_TRAIN_SCALE_COLORS = {
    "nsm":  ["#c9a3e8", "#8e44ad", "#4a1d6e"],   # light -> dark purple
    "unet": ["#fdba74", "#ea580c", "#7c2d12"],   # light -> dark orange
}

# --- Palette 2: per-model, single-scale conditions vs mixed-scale combos (plot_mixed_scale_by_model) ---
# blue/green family = single-scale training, red/orange family = mixed-scale training
SINGLE_SCALE_PALETTE = ["#3182bd", "#31a354", "#66c2a4", "#2c7fb8"]   # blues/greens
MIXED_SCALE_PALETTE  = ["#e6550d", "#de2d26", "#a63603", "#fd8d3c"]   # reds/oranges

# --- Palette 3: mixed-scale combos, both models on one plot (plot_mixed_vs_models) ---
# NSM = blue/green shades, UNet = red/orange/brown shades
NSM_MIXED_COLORS  = ["#3182bd", "#31a354", "#66c2a4", "#2c7fb8"]
UNET_MIXED_COLORS = ["#de2d26", "#e6550d", "#a63603", "#fd8d3c"]


def load_agg(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def discover_metrics(*aggs) -> list:
    """Discover metric names (keys ending in '_mean') across one or more agg dicts."""
    metrics = set()
    for agg in aggs:
        for model_type in agg.values():
            for train_key in model_type.values():
                for entry in train_key.values():
                    for key in entry:
                        if key.endswith("_mean"):
                            metrics.add(key[: -len("_mean")])
    return sorted(metrics)


def prettify_combo(combo_name: str) -> str:
    """'combined_8_4_2' -> 'L=0.8+0.4+0.2'"""
    if not combo_name.startswith("combined_"):
        return combo_name
    digits = combo_name.split("_")[1:]
    scales = [f"{int(d) / 10:g}" for d in digits]
    return "L=" + "+".join(scales)


# ---------------------------------------------------------------------------
# Shared generic bar-drawing + figure-building helpers
# ---------------------------------------------------------------------------

def _grouped_bar_subplot(ax, series, metric, test_scales):
    """
    Draw grouped bars on one axis for a list of `series`, grouped by test_scale.

    series: list of dicts, each {"label": str, "color": str, "data": dict}
        where data[test_scale] = entry dict containing f"{metric}_mean" /
        f"{metric}_std" / "in_dist".
    """
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


def _plot_metric_grid(series, metrics, test_scales, out_path, title, ncols=None):
    """
    Build a full figure: one subplot per metric, each drawn via
    `_grouped_bar_subplot` using the same `series` list across all metrics.
    """
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
        loc="upper center", ncol=min(len(labels) + 1, 4), bbox_to_anchor=(0.5, 1.15),
        frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.suptitle(title, y=1.25, fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 1. Single-scale training: all train scales x both models
# ---------------------------------------------------------------------------

def plot_all_scales_eval(
    agg: dict,
    output_dir: Path,
    metrics: list = None,
    train_scales: list = None,
    test_scales: list = None,
    model_types: list = None,
    title: str = None,
    ncols: int = None,
):
    """
    One subplot per metric. Each subplot groups bars by test scale, with
    (train_scale x model_type) combos side by side. Color families
    distinguish model (purple = NSM, orange = UNet); shade encodes train
    scale.
    """
    model_types = model_types or MODEL_TYPES
    metrics = metrics or discover_metrics(agg)
    train_scales = train_scales or sorted(
        set().union(*[agg.get(mt, {}).keys() for mt in model_types])
    )
    test_scales = test_scales or sorted(
        set().union(*[
            agg.get(mt, {}).get(ts, {}).keys()
            for mt in model_types for ts in train_scales
        ])
    )
    title = title or "Cross-scale eval: NSM vs UNet (all train scales)"

    series = []
    for ti, train_scale in enumerate(train_scales):
        for model_type in model_types:
            shades = MODEL_TRAIN_SCALE_COLORS.get(model_type, ["#888888"] * len(train_scales))
            series.append({
                "label": f"{model_type.upper()} (train {train_scale})",
                "color": shades[ti % len(shades)],
                "data": agg.get(model_type, {}).get(train_scale, {}),
            })

    out_path = Path(output_dir) / "all_scales_eval.png"
    return _plot_metric_grid(series, metrics, test_scales, out_path, title, ncols)


# ---------------------------------------------------------------------------
# 2. Per model: single-scale conditions + mixed-scale combos, same model only
# ---------------------------------------------------------------------------

def plot_mixed_scale_by_model(
    model_type: str,
    single_agg: dict,
    mixed_agg: dict,
    output_dir: Path,
    metrics: list = None,
    single_scales: list = None,
    combos: list = None,
    test_scales: list = None,
    title: str = None,
    ncols: int = None,
):
    """
    For a single model, plot ALL training conditions -- single-scale training
    runs AND mixed-scale combos -- as separate bar series, evaluated across
    all test scales. One subplot per metric.

    single_agg: agg[model_type][train_scale][test_scale] = entry   (e.g. cross_scale_summary.json)
    mixed_agg:  agg[model_type][combo][test_scale] = entry         (e.g. mixed_scale_summary.json)
    """
    single_scales = single_scales or sorted(single_agg.get(model_type, {}).keys())
    combos = combos or sorted(mixed_agg.get(model_type, {}).keys())
    metrics = metrics or discover_metrics(single_agg, mixed_agg)

    test_scales = test_scales or sorted(set().union(
        *[single_agg.get(model_type, {}).get(ts, {}).keys() for ts in single_scales],
        *[mixed_agg.get(model_type, {}).get(c, {}).keys() for c in combos],
    ))

    title = title or f"{model_type.upper()}: single-scale vs mixed-scale training"

    series = []
    for i, train_scale in enumerate(single_scales):
        series.append({
            "label": f"train {train_scale}",
            "color": SINGLE_SCALE_PALETTE[i % len(SINGLE_SCALE_PALETTE)],
            "data": single_agg.get(model_type, {}).get(train_scale, {}),
        })
    for i, combo in enumerate(combos):
        series.append({
            "label": f"train {prettify_combo(combo)}",
            "color": MIXED_SCALE_PALETTE[i % len(MIXED_SCALE_PALETTE)],
            "data": mixed_agg.get(model_type, {}).get(combo, {}),
        })

    out_path = Path(output_dir) / f"mixed_scale_by_model_{model_type}.png"
    return _plot_metric_grid(series, metrics, test_scales, out_path, title, ncols)


# ---------------------------------------------------------------------------
# 3. Mixed-scale combos only, both models on the same plot
# ---------------------------------------------------------------------------

def plot_mixed_vs_models(
    mixed_agg: dict,
    output_dir: Path,
    metrics: list = None,
    combos: list = None,
    test_scales: list = None,
    model_types: list = None,
    title: str = None,
    ncols: int = None,
):
    """
    Mixed-scale training combos ONLY (no single-scale conditions), both
    models on the same plot. One subplot per metric. Color families
    distinguish model (blue/green = NSM, red/orange = UNet); shade encodes
    combo.
    """
    model_types = model_types or MODEL_TYPES
    combos = combos or sorted(
        set().union(*[mixed_agg.get(mt, {}).keys() for mt in model_types])
    )
    metrics = metrics or discover_metrics(mixed_agg)
    test_scales = test_scales or sorted(
        set().union(*[
            mixed_agg.get(mt, {}).get(c, {}).keys()
            for mt in model_types for c in combos
        ])
    )
    title = title or "Mixed-scale training: NSM vs UNet"

    model_palettes = {"nsm": NSM_MIXED_COLORS, "unet": UNET_MIXED_COLORS}

    series = []
    for i, combo in enumerate(combos):
        for model_type in model_types:
            palette = model_palettes.get(model_type, ["#888888"] * len(combos))
            series.append({
                "label": f"{model_type.upper()} ({prettify_combo(combo)})",
                "color": palette[i % len(palette)],
                "data": mixed_agg.get(model_type, {}).get(combo, {}),
            })

    out_path = Path(output_dir) / "mixed_scale_combined_vs_models.png"
    return _plot_metric_grid(series, metrics, test_scales, out_path, title, ncols)


if __name__ == "__main__":
    single_agg = load_agg(Path("eval_results_cross/cross_scale_summary.json"))
    mixed_agg = load_agg(Path("eval_results_mixed_scale_training/mixed_scale_summary.json"))
    out_dir = Path("eval_results_mixed_scale_training")

    # plot_all_scales_eval(single_agg, output_dir=Path("eval_results_cross"))

    for model_type in MODEL_TYPES:
        plot_mixed_scale_by_model(model_type, single_agg, mixed_agg, output_dir=out_dir)

    plot_mixed_vs_models(mixed_agg, output_dir=out_dir)