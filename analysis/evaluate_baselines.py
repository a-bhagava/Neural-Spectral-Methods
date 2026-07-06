# import argparse
# import csv
# import json
# from pathlib import Path
# import os
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import numpy as _np
# import ast

# from src.pde.navierstokes import re4_2, re4_4, re4_8, re4_1, re4_3, re4_6, re4_10
# from src import *
# from src.pde import *
# from src.model import *
# from src.basis import *
# from src.train import eval as train_eval


# NSM_ROOT_DIRS = {
#     "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_2"),
#     "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_4"),
#     "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_8"),
# }

# UNET_ROOT_DIRS = {
#     "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_2"),
#     "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_4"),
#     "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_8"),
# }

# IN_DIST_PDES = {
#     "L=0.2": re4_2,
#     "L=0.4": re4_4,
#     "L=0.8": re4_8,
# }

# OUT_OF_DIST_PDES = {
#     "L=0.1": re4_1, 
#     "L=0.3": re4_3,
#     "L=0.6": re4_6,
#     "L=1.0": re4_10,
# }


# # ── loading ───────────────────────────────────────────────────────────────────

# def load_cfg(run_dir: Path) -> dict:
#     cfg_path = run_dir / "cfg"
#     if not cfg_path.exists():
#         raise FileNotFoundError(f"No cfg found in {run_dir}")
#     return ast.literal_eval(cfg_path.read_text())


# def load_model_and_variable(run_dir: Path, pde):
#     """
#     Mirrors the loading pattern in main.py:
#         variable = np.load(cfg["load"], allow_pickle=True).item()
#         model    = model.bind(variable, rngs=next(rngs))
#     """
#     from importlib import import_module
#     import jax.random as random

#     cfg      = load_cfg(run_dir)
#     variable = np.load(run_dir / "variable_ckpt.npy", allow_pickle=True).item()

#     if cfg.get("f64"):
#         jax.config.update("jax_enable_x64", True)

#     # reconstruct model — mirrors main.py model loading block
#     col = cfg["model"]
#     if cfg.get("spectral"):     col += ".spectral"
#     elif cfg.get("multiscale"): col += ".multiscale"
#     elif cfg.get("hierarchical"): col += ".hierarchical"

#     mod   = import_module(f"src.model.{col}")
#     Model = getattr(mod, cfg["model"].upper())
#     model = Model(pde, cfg)

#     prng  = random.PRNGKey(cfg.get("seed", 0))
#     rngs  = RNGS(prng, ["params", "sample"])
#     bound = model.bind(variable, rngs=next(rngs))

#     return bound, variable, cfg


# # ── evaluation ────────────────────────────────────────────────────────────────

# def run_eval(model, variable, test_pde, cfg):
#     """
#     Mirrors the eval() call in main.py, applied to a test PDE directly.
#     Uses the same metric() function your Trainer uses internally.
#     """

#     train = Trainer(mod=model, pde=test_pde, cfg=cfg)
#     prng = jax.random.PRNGKey(cfg.get("seed", 0))
#     rngs = RNGS(prng, ["params", "sample"])

#     metrics, predictions = train.apply(
#         {}, # stateless test-time evaluation, no state to pass
#         variable,
#         method=train_eval,
#         rngs=next(rngs),
#     )
#     u_true, u_pred = predictions
#     return metrics, u_true, u_pred

# # ── main loop ─────────────────────────────────────────────────────────────────

# # evaluate each seed inside the model dict across the different test scales 
# # save the predictions inside each of the directories for each test scale
# # save the mean and std of the metrics across seeds for each test scale 
# # save the spectral metrics 
# # save the mean / stdev of the spectral metrics across seeds for each test scale 
# # generate bar charts for the metrics across test scales for each model type (nsm, unet)
# # save a snapshot of the predictions for each test scale for each model type (nsm, unet) -- for one seed as an example
# # make plots based on the spectral metric plot, but evaluated on all length scales 
# # side-by-side comparison of the two architectures (with error bars on the metrics)

"""
Cross-scale evaluation for NSM and UNet models.

Loads models trained on individual length scales (L=0.2, 0.4, 0.8), each with
multiple random seeds, and evaluates every seed against every test length
scale (the in-distribution scales plus out-of-distribution scales). Produces:

  - per-seed predictions + metrics on disk
  - a mean/std aggregation across seeds for every (model_type, train_scale, test_scale)
  - one representative snapshot plot per (train_scale, test_scale) pair
  - grouped bar charts comparing NSM vs UNet, faceted by training scale,
    with a star marking the in-distribution test scale

Usage:
    python -m analysis.evaluate_baselines --output-dir eval_results_cross
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import ast
import jax
import jax.numpy as jnp
import numpy as _np
import matplotlib.pyplot as plt

from src.pde.navierstokes import re4_2, re4_4, re4_8, re4_1, re4_3, re4_6, re4_10
from src import *
from src.pde import *
from src.model import *
from src.basis import *
from src.train import eval as train_eval


# ── config ───────────────────────────────────────────────────────────────────

NSM_ROOT_DIRS = {
    "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_2"),
    "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_4"),
    "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_8"),
}

# "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_2"),
# "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_4"),

UNET_ROOT_DIRS = {
    "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_8"),
}

# model_type -> {train_scale -> root_dir_containing_seed_subdirs}
# "nsm":  NSM_ROOT_DIRS,
MODEL_ROOT_DIRS = {
    "unet": UNET_ROOT_DIRS,
}

IN_DIST_PDES = {
    "L=0.2": re4_2,
    "L=0.4": re4_4,
    "L=0.8": re4_8,
}

OUT_OF_DIST_PDES = {
    "L=0.1": re4_1,
    "L=0.3": re4_3,
    "L=0.6": re4_6,
    "L=1.0": re4_10,
}

ALL_TEST_PDES = {**IN_DIST_PDES, **OUT_OF_DIST_PDES}

TRAIN_SCALES = list(IN_DIST_PDES.keys())      # ["L=0.2", "L=0.4", "L=0.8"] — x-axis in the headline chart
TEST_SCALES  = list(ALL_TEST_PDES.keys())     # includes OOD scales, for the extended chart
MODEL_TYPES  = list(MODEL_ROOT_DIRS.keys())   # ["nsm", "unet"]

# Metric key for the headline bar chart in your reference image.
# Change this if your metrics dict uses a different key for the spectral distance.
SPECTRAL_METRIC_KEY = "ralsd"


# ── loading ──────────────────────────────────────────────────────────────────

def load_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No cfg found in {run_dir}")
    return ast.literal_eval(cfg_path.read_text())


def find_seed_runs(root_dir: Path) -> list[Path]:
    """Every subdirectory of root_dir that looks like a completed run (one per seed)."""
    if not root_dir.exists():
        return []
    runs = []
    for run_dir in sorted(root_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "variable_ckpt.npy").exists() and (run_dir / "cfg").exists():
            runs.append(run_dir)
    return runs


def load_model_and_variable(run_dir: Path, pde):
    """Mirrors the loading pattern in main.py."""
    from importlib import import_module
    import jax.random as random

    cfg      = load_cfg(run_dir)
    variable = np.load(run_dir / "variable_ckpt.npy", allow_pickle=True).item()

    if cfg.get("f64"):
        jax.config.update("jax_enable_x64", True)

    col = cfg["model"]
    if cfg.get("spectral"):       col += ".spectral"
    elif cfg.get("multiscale"):   col += ".multiscale"
    elif cfg.get("hierarchical"): col += ".hierarchical"

    mod   = import_module(f"src.model.{col}")
    Model = getattr(mod, cfg["model"].upper())
    model = Model(pde, cfg)

    prng  = random.PRNGKey(cfg.get("seed", 0))
    rngs  = RNGS(prng, ["params", "sample"])
    bound = model.bind(variable, rngs=next(rngs))

    return bound, variable, cfg


# ── evaluation ───────────────────────────────────────────────────────────────

def run_eval(model, variable, test_pde, cfg):
    train = Trainer(mod=model, pde=test_pde, cfg=cfg)
    prng  = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs  = RNGS(prng, ["params", "sample"])

    metrics, predictions = train.apply(
        {},
        variable,
        method=train_eval,
        rngs=next(rngs),
    )
    u_true, u_pred = predictions
    return metrics, u_true, u_pred


# ── plotting: single prediction snapshot ────────────────────────────────────

def plot_snapshot(u_true, u_pred, out_path: Path, title: str):
    def _extract(arr):
        arr = np.asarray(arr)
        if arr.ndim == 5: return arr[0, -1, ..., 0]
        if arr.ndim == 4: return arr[0, ..., 0]
        return arr

    true_f  = _extract(u_true)
    pred_f  = _extract(u_pred)
    error_f = np.abs(true_f - pred_f)
    vmax    = max(np.abs(true_f).max(), np.abs(pred_f).max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, label, cmap in zip(
        axes,
        [true_f, pred_f, error_f],
        ["Ground truth", "Prediction", "Absolute error"],
        ["RdBu_r", "RdBu_r", "inferno"],
    ):
        kw = dict(origin="lower", cmap=cmap)
        if label != "Absolute error":
            kw.update(vmin=-vmax, vmax=vmax)
        im = ax.imshow(field, **kw)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main evaluation loop ─────────────────────────────────────────────────────

def evaluate_all(output_dir: Path) -> list[dict]:
    """
    For every (model_type, train_scale) pair, evaluates every seed run against
    every test scale (in-dist + OOD). Saves per-seed predictions/metrics to
    disk, plus one representative snapshot per (train_scale, test_scale) using
    the first seed found. Returns a flat list of per-seed result rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_type, root_dirs in MODEL_ROOT_DIRS.items():
        for train_scale, root_dir in root_dirs.items():
            seed_runs = find_seed_runs(root_dir)
            if not seed_runs:
                print(f"  [skip] no seed runs found for {model_type}/{train_scale} in {root_dir}")
                continue
            print(f"\n{model_type} / train {train_scale}: {len(seed_runs)} seed run(s)")

            train_pde = IN_DIST_PDES[train_scale]

            for seed_idx, run_dir in enumerate(seed_runs):
                print(f"  seed run: {run_dir.name}")
                try:
                    model, variable, cfg = load_model_and_variable(run_dir, train_pde)
                except Exception as e:
                    print(f"    FAILED to load: {e}")
                    continue

                for test_scale, test_pde in ALL_TEST_PDES.items():
                    in_dist = (test_scale == train_scale)
                    print(f"    -> {test_scale} ... ", end="", flush=True)

                    out_dir = (output_dir / model_type / f"train_{train_scale}"
                               / f"test_{test_scale}" / run_dir.name)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        metrics, u_true, u_pred = run_eval(model, variable, test_pde, cfg)

                        # _np.savez_compressed(
                        #     out_dir / "eval.npz",
                        #     u_pred=_np.array(u_pred),
                        #     u_true=_np.array(u_true),
                        # )
                        with (out_dir / "metrics.json").open("w") as f:
                            json.dump(
                                {k: (float(v) if v is not None else None) for k, v in metrics.items()},
                                f, indent=2,
                            )

                        # Only render a snapshot for the first seed of each
                        # (train_scale, test_scale) pair -- one example is enough.
                        if seed_idx == 0:
                            plot_snapshot(
                                u_true, u_pred,
                                out_path=out_dir / "snapshot.png",
                                title=(f"{model_type} trained {train_scale} -> test {test_scale}"
                                       + ("  [in-dist]" if in_dist else "  [OOD]")),
                            )

                        row = {
                            "model_type": model_type,
                            "train_scale": train_scale,
                            "test_scale": test_scale,
                            "seed_run": run_dir.name,
                            "in_dist": in_dist,
                            "status": "ok",
                            **{k: (float(v) if v is not None else None) for k, v in metrics.items()},
                        }
                        rows.append(row)

                        extra = (f"  {SPECTRAL_METRIC_KEY}={metrics[SPECTRAL_METRIC_KEY]:.4f}"
                                 if SPECTRAL_METRIC_KEY in metrics else "")
                        print(f"errr={metrics.get('errr', float('nan')):.4f}{extra}"
                              + ("  [in-dist]" if in_dist else ""))

                    except Exception as e:
                        print(f"FAILED: {e}")
                        rows.append({
                            "model_type": model_type,
                            "train_scale": train_scale,
                            "test_scale": test_scale,
                            "seed_run": run_dir.name,
                            "in_dist": in_dist,
                            "status": f"failed: {e}",
                        })

    if rows:
        raw_path = output_dir / "cross_scale_raw.csv"
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with raw_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRaw per-seed results -> {raw_path}")

    return rows


# ── aggregation across seeds ─────────────────────────────────────────────────

def aggregate_across_seeds(rows: list[dict], output_dir: Path) -> dict:
    """
    Groups per-seed rows by (model_type, train_scale, test_scale) and computes
    mean/std for every numeric metric found. Returns a nested dict:

        agg[model_type][train_scale][test_scale] = {
            "in_dist": bool, "n_seeds": int,
            "<metric>_mean": float, "<metric>_std": float, ...
        }

    Also writes a flat summary CSV and a JSON dump of the nested structure.
    """
    groups = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["model_type"], r["train_scale"], r["test_scale"])
        groups[key].append(r)

    skip_keys = {"model_type", "train_scale", "test_scale", "seed_run", "in_dist", "status"}
    metric_keys = sorted({k for r in rows for k in r.keys() if k not in skip_keys})

    agg = defaultdict(lambda: defaultdict(dict))
    summary_rows = []

    for (model_type, train_scale, test_scale), group in groups.items():
        entry = {"in_dist": group[0]["in_dist"], "n_seeds": len(group)}
        for mk in metric_keys:
            vals = _np.array([r[mk] for r in group if r.get(mk) is not None], dtype=float)
            if vals.size == 0:
                continue
            entry[f"{mk}_mean"] = float(vals.mean())
            entry[f"{mk}_std"]  = float(vals.std())
        agg[model_type][train_scale][test_scale] = entry

        summary_rows.append({
            "model_type": model_type,
            "train_scale": train_scale,
            "test_scale": test_scale,
            **entry,
        })

    if summary_rows:
        summary_path = output_dir / "cross_scale_summary.csv"
        fieldnames = sorted({k for r in summary_rows for k in r.keys()})
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Aggregated (mean/std across seeds) -> {summary_path}")

    with (output_dir / "cross_scale_summary.json").open("w") as f:
        json.dump(agg, f, indent=2)

    return agg


# ── plotting: grouped bar chart, faceted by train scale ─────────────────────

def plot_cross_scale_bars(
    agg: dict,
    metric: str,
    output_dir: Path,
    test_scales: list = None,
    train_scales: list = None,
    title: str = None,
):
    """
    Reproduces the "Cross-scale <METRIC>: NSM vs UNet" figure: one subplot per
    training scale, grouped bars (NSM vs UNet) for each test scale, error bars
    from the std across seeds, and a star marking the in-distribution bar.
    """
    test_scales  = test_scales  or TRAIN_SCALES   # default matches the reference image (in-dist scales only)
    train_scales = train_scales or TRAIN_SCALES
    model_colors = {"nsm": "#1f77b4", "unet": "#ff7f0e"}
    metric_label = metric.upper() if len(metric) <= 6 else metric.replace("_", " ").title()
    title = title or f"Cross-scale {metric_label}: NSM vs UNet"

    fig, axes = plt.subplots(
        1, len(train_scales), figsize=(5.5 * len(train_scales), 4.5), sharey=True
    )
    if len(train_scales) == 1:
        axes = [axes]

    bar_width = 0.35
    x = _np.arange(len(test_scales))

    for ax, train_scale in zip(axes, train_scales):
        for i, model_type in enumerate(MODEL_TYPES):
            means, stds, in_dists = [], [], []
            for test_scale in test_scales:
                entry = agg.get(model_type, {}).get(train_scale, {}).get(test_scale)
                if entry is None or f"{metric}_mean" not in entry:
                    means.append(_np.nan); stds.append(0.0); in_dists.append(False)
                else:
                    means.append(entry[f"{metric}_mean"])
                    stds.append(entry.get(f"{metric}_std", 0.0))
                    in_dists.append(entry["in_dist"])

            offset = (i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=stds, capsize=3,
                label=model_type.upper(),
                color=model_colors[model_type],
            )
            # NSM: solid outline. UNet: dashed outline (matches reference image).
            edge_style = "--" if model_type == "unet" else "-"
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.2)
                b.set_linestyle(edge_style)

            for xi, mean, std, in_dist in zip(x + offset, means, stds, in_dists):
                if in_dist and not _np.isnan(mean):
                    ax.annotate("*", (xi, mean + std), ha="center", va="bottom", fontsize=14)

        ax.set_title(f"Train {train_scale}")
        ax.set_xticks(x)
        ax.set_xticklabels(test_scales)
        ax.set_xlabel("Test scale")

    axes[0].set_ylabel(f"{metric_label} (lower = better)")

    handles, labels = axes[0].get_legend_handles_labels()
    star_handle = plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10)
    fig.legend(
        handles + [star_handle], labels + ["in-distribution"],
        loc="upper center", ncol=len(labels) + 1, bbox_to_anchor=(0.5, 1.05), frameon=False,
    )
    fig.suptitle(title, y=1.15, fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"cross_scale_{metric}_bars.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart ({metric}) -> {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-scale NSM vs UNet evaluation")
    parser.add_argument("--output-dir", default="eval_results_cross")
    parser.add_argument(
        "--bar-metrics", nargs="+", default=["errr", SPECTRAL_METRIC_KEY],
        help="Metric keys to render as grouped bar charts (one figure per metric).",
    )
    parser.add_argument(
        "--include-ood-in-bars", action="store_true",
        help="Also plot OOD test scales (L=0.1, 0.3, 0.6, 1.0) alongside the in-dist ones.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = evaluate_all(output_dir)
    agg  = aggregate_across_seeds(rows, output_dir)

    bar_test_scales = TEST_SCALES if args.include_ood_in_bars else TRAIN_SCALES
    for metric in args.bar_metrics:
        plot_cross_scale_bars(agg, metric, output_dir, test_scales=bar_test_scales)

