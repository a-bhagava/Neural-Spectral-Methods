# """
# Cross-scale evaluation for NSM models.

# Tests each model checkpoint against every length-scale dataset,
# producing a summary CSV and per-pair snapshots + metrics.

# Usage:
#     python cross_scale_eval.py \
#         --model-dirs  path/to/model_L0.2  path/to/model_L0.4  path/to/model_L0.8 \
#         --output-dir  eval_results

# The script infers the PDE from each model's saved cfg, so you don't need
# to pass dataset paths separately — it loads the solution from the PDE class
# directly, exactly as training does.
# """

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as _np
import ast

from src.pde.navierstokes import re4_2, re4_4, re4_8
from src import *
from src.pde import *
from src.model import *
from src.basis import *

# ── config ────────────────────────────────────────────────────────────────────

# "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/nsm_hyperparam/nsT3re4.length0.2"),
# "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/nsm_hyperparam/nsT3re4.length0.4"),
# "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/nsm_hyperparam/nsT3re4.length0.8"),
# "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/wav_hyperparam/nsT3re4.length0.2"),


ROOT_DIRS = {
    "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/unet_hyperparam/nsT3re4.length0.2"),
    "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/unet_hyperparam/nsT3re4.length0.4"),
    "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/unet_hyperparam/nsT3re4.length0.8"),
}

PDES = {
    "L=0.2": re4_2,
    "L=0.4": re4_4,
    "L=0.8": re4_8,
}

SCALES = list(ROOT_DIRS.keys())   # ["L=0.2", "L=0.4", "L=0.8"]


# ── find best run (mirrors your notebook logic) ───────────────────────────────

def find_best_run(root_dir: Path) -> tuple[Path, dict]:
    """
    Scan subdirectories of root_dir, read metric.residual.npy from each,
    and return the path + metrics of the run with the lowest final residual.
    """
    best_path, best_metrics, best_residual = None, None, float("inf")

    for run_dir in sorted(root_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_path     = run_dir / "variable_ckpt.npy"
        residual_path = run_dir / "metric.residual.npy"
        if not ckpt_path.exists() or not residual_path.exists():
            continue
        try:
            residual_curve = np.load(residual_path)
            final_residual = float(residual_curve[-1])

            m = {"residual": final_residual}
            for key in ("errr", "erra", "rate"):
                p = run_dir / f"metric.{key}.npy"
                if p.exists():
                    arr = np.load(p)
                    m[key] = float(arr[-1])

            if final_residual < best_residual:
                best_residual = final_residual
                best_path     = run_dir
                best_metrics  = m

        except Exception as e:
            print(f"  Warning: could not read {run_dir.name}: {e}")

    if best_path is None:
        raise RuntimeError(f"No valid runs found in {root_dir}")

    print(f"  Best run : {best_path.name}")
    print(f"  Residual : {best_residual:.6f}")
    if "errr" in best_metrics:
        print(f"  Rel L2   : {best_metrics['errr']:.6f}")
    return best_path, best_metrics


# ── loading ───────────────────────────────────────────────────────────────────

def load_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No cfg found in {run_dir}")
    return ast.literal_eval(cfg_path.read_text())


def load_model_and_variable(run_dir: Path, pde):
    """
    Mirrors the loading pattern in main.py:
        variable = np.load(cfg["load"], allow_pickle=True).item()
        model    = model.bind(variable, rngs=next(rngs))
    """
    from importlib import import_module
    import jax.random as random

    cfg      = load_cfg(run_dir)
    variable = np.load(run_dir / "variable_ckpt.npy", allow_pickle=True).item()

    if cfg.get("f64"):
        jax.config.update("jax_enable_x64", True)

    # reconstruct model — mirrors main.py model loading block
    col = cfg["model"]
    if cfg.get("spectral"):     col += ".spectral"
    elif cfg.get("multiscale"): col += ".multiscale"
    elif cfg.get("hierarchical"): col += ".hierarchical"

    mod   = import_module(f"src.model.{col}")
    Model = getattr(mod, cfg["model"].upper())
    model = Model(pde, cfg)

    prng  = random.PRNGKey(cfg.get("seed", 0))
    rngs  = RNGS(prng, ["params", "sample"])
    bound = model.bind(variable, rngs=next(rngs))

    return bound, variable, cfg


# ── evaluation ────────────────────────────────────────────────────────────────

def run_eval(model, variable, test_pde, cfg):
    """
    Mirrors the eval() call in main.py, applied to a test PDE directly.
    Uses the same metric() function your Trainer uses internally.
    """
    from src.train import eval as train_eval

    # build a minimal trainer-like object with the test pde
    train = Trainer(mod=model, pde=test_pde, cfg=cfg)

    prng = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs = RNGS(prng, ["params", "sample"])

    metrics, predictions = train.apply(
        {},          # no mutable state needed for eval
        variable,
        method=train_eval,
        rngs=next(rngs),
    )
    u_true, u_pred = predictions
    return metrics, u_true, u_pred


# ── plotting ──────────────────────────────────────────────────────────────────

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
        [true_f,    pred_f,    error_f],
        ["Ground truth", "Prediction", "Absolute error"],
        ["RdBu_r",  "RdBu_r",  "inferno"],
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


# ── main loop ─────────────────────────────────────────────────────────────────

def evaluate_all(output_dir: Path, metric_key: str = "errr"):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Step 1: find best run per training scale
    best_runs = {}
    for scale, root_dir in ROOT_DIRS.items():
        print(f"\nFinding best run for train scale {scale}:")
        run_dir, _ = find_best_run(root_dir)
        best_runs[scale] = run_dir

    # Step 2: load each best model once, evaluate across all test scales
    for train_scale, run_dir in best_runs.items():
        train_pde = PDES[train_scale]
        print(f"\nLoading model trained on {train_scale}: {run_dir.name}")
        bound_model, variable, cfg = load_model_and_variable(run_dir, train_pde)

        for test_scale, test_pde in PDES.items():
            print(f"  → Testing on {test_scale} ... ", end="", flush=True)
            try:
                metrics, u_true, u_pred = run_eval(bound_model, variable, test_pde, cfg)

                # mark OOD
                ood = train_scale != test_scale

                print(f"erra={metrics['erra']:.4f}  "
                      f"errr={metrics['errr']:.4f}  "
                      f"residual={metrics.get('residual', float('nan')):.4f}"
                      + ("  [OOD]" if ood else "  [in-dist]"))

                # save outputs
                pair_dir = output_dir / f"train_{train_scale}" / f"test_{test_scale}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                _np.savez_compressed(pair_dir / "eval.npz",
                                    u_pred=_np.array(u_pred),
                                    u_true=_np.array(u_true))
                with (pair_dir / "metrics.json").open("w") as f:
                    json.dump({k: (float(v) if v is not None else None)
                               for k, v in metrics.items()}, f, indent=2)

                plot_snapshot(u_true, u_pred,
                              out_path=pair_dir / "snapshot.png",
                              title=(f"Train {train_scale} → Test {test_scale}"
                                     f"  |  errr={metrics['errr']:.4f}"
                                     + ("  [OOD]" if ood else "")))

                rows.append({
                    "model":       run_dir.name,
                    "train_scale": train_scale,
                    "test_scale":  test_scale,
                    "ood":         ood,
                    "status":      "ok",
                    **{k: (float(v) if v is not None else None)
                       for k, v in metrics.items()},
                })

            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({
                    "model":       run_dir.name,
                    "train_scale": train_scale,
                    "test_scale":  test_scale,
                    "ood":         train_scale != test_scale,
                    "status":      f"failed: {e}",
                })

    # save summary CSV
    if rows:
        summary_path = output_dir / "cross_scale_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary → {summary_path}")

    return rows


# ── 3x3 table printer ─────────────────────────────────────────────────────────

def print_table(rows: list, metric: str = "errr"):
    lookup = {}
    for r in rows:
        if r.get("status") == "ok" and metric in r:
            lookup[(r["train_scale"], r["test_scale"])] = r[metric]

    col_w = 10
    col_header = "train \\ test"
    header = f"{col_header:<14}" + "".join(f"{s:>{col_w}}" for s in SCALES)
    print(f"\n{metric} — cross-scale table  (diagonal = in-distribution):")
    print(header)
    print("─" * len(header))
    for train in SCALES:
        row = f"{train:<14}"
        for test in SCALES:
            val = lookup.get((train, test))
            marker = "" if train != test else "*"
            row += f"{f'{val:.4f}{marker}':>{col_w}}" if val is not None else f"{'—':>{col_w}}"
        print(row)
    print("  * = in-distribution")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-scale NSM evaluation")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--metric", default="errr",
                        choices=["errr", "erra", "residual"])
    args = parser.parse_args()

    rows = evaluate_all(Path(args.output_dir), args.metric)
    print_table(rows, args.metric)