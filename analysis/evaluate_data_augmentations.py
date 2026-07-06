"""
Data-augmentation / dropout analysis for NSM (trained on L=0.8 only).

Run directory naming convention, e.g.:
    energy_std1.0_dr0.0_hdim32_depth15_mode12_41_41_lr0.001_42
    uniform_std0.01_dr0.0_hdim32_depth15_mode12_41_41_lr0.001_42

Produces:
  1. sigma_sweep.png    -- dr=0.0, increasing sigma, uniform vs energy (residual curves)
  2. dropout_sweep.png  -- fixed best sigma (picked per noise type from plot 1),
                           increasing dropout, uniform vs energy (residual curves)
  3. Cross-scale evaluation of the single best run in the whole folder
     (lowest final residual, via find_best_run) against every test scale.

Usage:
    python -m analysis.data_augmentation_analysis --output-dir eval_results_data_aug
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import ast
import jax
import numpy as _np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pde.navierstokes import re4_1, re4_2, re4_3, re4_4, re4_6, re4_8, re4_10
from src import *
from src.pde import *
from src.model import *
from src.basis import *
from src.train import eval as train_eval


# ── config ───────────────────────────────────────────────────────────────────

ROOT_DIR = Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/data_augmentation/nsm/re4_8")

NOISE_TYPES  = ["uniform", "energy"]
NOISE_LABELS = {"uniform": "Uniform noise", "energy": "Energy-scaled noise"}

STDS      = [0.01, 0.1, 1.0]
DROPOUTS  = [0.0, 0.1, 0.2]

TRAIN_PDE = re4_8   # only trained on L=0.8

TEST_PDES = {
    "L=0.1": re4_1,
    "L=0.2": re4_2,
    "L=0.3": re4_3,
    "L=0.4": re4_4,
    "L=0.6": re4_6,
    "L=0.8": re4_8,
    "L=1.0": re4_10,
}
TRAIN_TEST_SCALE = "L=0.8"

RUN_NAME_RE = re.compile(
    r"^(?P<noise>uniform|energy)_std(?P<std>[\d.]+)_dr(?P<dr>[\d.]+)_"
    r"hdim(?P<hdim>\d+)_depth(?P<depth>\d+)_mode(?P<mode>\d+)_"
    r"(?P<m1>\d+)_(?P<m2>\d+)_lr(?P<lr>[\d.]+)_(?P<seed>\d+)$"
)


# ── run discovery ────────────────────────────────────────────────────────────

def parse_run_name(name: str) -> dict | None:
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "noise": d["noise"],
        "std":   float(d["std"]),
        "dr":    float(d["dr"]),
        "seed":  d["seed"],
    }


def discover_runs(root_dir: Path) -> list[dict]:
    """Every subdirectory of root_dir matching the run-name convention."""
    runs = []
    for run_dir in sorted(root_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        parsed["path"] = run_dir
        runs.append(parsed)
    if not runs:
        print(f"  [warn] no runs matched the naming convention in {root_dir}")
    return runs


def load_residual_curve(run_dir: Path):
    p = run_dir / "metric.residual.npy"
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception as e:
        print(f"    [warn] could not load {p}: {e}")
        return None


def best_seed_in_group(group: list[dict]) -> dict | None:
    """Among runs sharing (noise, std, dr), pick the one with lowest final residual."""
    best, best_val = None, float("inf")
    for r in group:
        curve = load_residual_curve(r["path"])
        if curve is None or len(curve) == 0:
            continue
        final = float(curve[-1])
        if final < best_val:
            best_val, best = final, r
    return best


def group_runs(runs: list[dict], noise: str, dr: float = None, std: float = None) -> list[dict]:
    out = [r for r in runs if r["noise"] == noise]
    if dr is not None:
        out = [r for r in out if abs(r["dr"] - dr) < 1e-9]
    if std is not None:
        out = [r for r in out if abs(r["std"] - std) < 1e-9]
    return out


# ── plot 1: sigma sweep at dr=0.0 ────────────────────────────────────────────

def plot_sigma_sweep(runs: list[dict], output_dir: Path) -> dict:
    """
    dr=0.0, increasing sigma, uniform vs energy. Returns best_sigma per noise
    type (lowest final residual) for use in the dropout sweep.
    """
    # One distinct, high-contrast color per (noise, sigma) curve -- 6 total.
    curve_colors = {
        ("uniform", 0.01): "#e6194B",
        ("uniform", 0.1):  "#3cb44b",
        ("uniform", 1.0):  "#4363d8",
        ("energy",  0.01): "#f58231",
        ("energy",  0.1):  "#911eb4",
        ("energy",  1.0):  "#42d4f4",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    best_sigma = {}

    for noise in NOISE_TYPES:
        best_std, best_val = None, float("inf")
        for std in STDS:
            group = group_runs(runs, noise=noise, dr=0.0, std=std)
            rep = best_seed_in_group(group)
            if rep is None:
                print(f"  [skip] no dr=0.0 run found for noise={noise} std={std}")
                continue
            curve = load_residual_curve(rep["path"])
            ax.plot(
                curve,
                color=curve_colors.get((noise, std), "#000000"),
                linewidth=2.4,
                label=f"{NOISE_LABELS[noise]}, sigma={std}",
                alpha=0.95,
            )
            final = float(curve[-1])
            if final < best_val:
                best_val, best_std = final, std
        if best_std is not None:
            best_sigma[noise] = best_std
            print(f"  best sigma for {noise}: {best_std}  (final residual={best_val:.6g})")

    ax.set_yscale("log")
    ax.set_xlabel("Training step (logged)", fontsize=12)
    ax.set_ylabel("PDE residual", fontsize=12)
    ax.set_title("Residual curves: sigma sweep (dropout = 0.0)", fontsize=13, fontweight="bold")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)
    ax.tick_params(labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9.5, ncol=1, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()

    out_path = output_dir / "sigma_sweep.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return best_sigma


# ── plot 2: dropout sweep at each noise type's best sigma ──────────────────

def plot_dropout_sweep(runs: list[dict], best_sigma: dict, output_dir: Path):
    # One distinct, high-contrast color per (noise, dropout) curve -- 6 total.
    curve_colors = {
        ("uniform", 0.0): "#e6194B",
        ("uniform", 0.1): "#3cb44b",
        ("uniform", 0.2): "#4363d8",
        ("energy",  0.0): "#f58231",
        ("energy",  0.1): "#911eb4",
        ("energy",  0.2): "#42d4f4",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for noise in NOISE_TYPES:
        std = best_sigma.get(noise)
        if std is None:
            print(f"  [skip] no best sigma found for {noise}, skipping in dropout sweep")
            continue
        for dr in DROPOUTS:
            group = group_runs(runs, noise=noise, dr=dr, std=std)
            rep = best_seed_in_group(group)
            if rep is None:
                print(f"  [skip] no run found for noise={noise} std={std} dr={dr}")
                continue
            curve = load_residual_curve(rep["path"])
            ax.plot(
                curve,
                color=curve_colors.get((noise, dr), "#000000"),
                linewidth=2.4,
                label=f"{NOISE_LABELS[noise]}, dropout={dr} (sigma={std})",
                alpha=0.95,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Training step (logged)", fontsize=12)
    ax.set_ylabel("PDE residual", fontsize=12)
    ax.set_title("Residual curves: dropout sweep (fixed best sigma per noise type)", fontsize=13, fontweight="bold")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)
    ax.tick_params(labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, ncol=1, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()

    out_path = output_dir / "dropout_sweep.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── find best run overall (mirrors mixed_scale_eval.py) ─────────────────────

def find_best_run(root_dir: Path) -> tuple[Path, dict]:
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
                    m[key] = float(np.load(p)[-1])

            if final_residual < best_residual:
                best_residual = final_residual
                best_path     = run_dir
                best_metrics  = m

        except Exception as e:
            print(f"  [warn] could not read {run_dir.name}: {e}")

    if best_path is None:
        raise RuntimeError(f"No valid runs found in {root_dir}")

    print(f"  Best run : {best_path.name}")
    print(f"  Residual : {best_residual:.6g}")
    if "errr" in best_metrics:
        print(f"  Rel L2   : {best_metrics['errr']:.6g}")
    return best_path, best_metrics


# ── loading + eval (mirrors cross_scale_eval.py) ─────────────────────────────

def load_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No cfg found in {run_dir}")
    return ast.literal_eval(cfg_path.read_text())


def load_model_and_variable(run_dir: Path, pde):
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


def run_eval(model, variable, test_pde, cfg):
    train = Trainer(mod=model, pde=test_pde, cfg=cfg)
    prng  = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs  = RNGS(prng, ["params", "sample"])

    metrics, predictions = train.apply(
        {}, variable, method=train_eval, rngs=next(rngs),
    )
    u_true, u_pred = predictions
    return metrics, u_true, u_pred


def evaluate_best_run_cross_scale(root_dir: Path, output_dir: Path) -> list[dict]:
    print(f"\nFinding best run in {root_dir}:")
    best_dir, _ = find_best_run(root_dir)

    parsed = parse_run_name(best_dir.name) or {}
    print(f"  -> noise={parsed.get('noise')} std={parsed.get('std')} dr={parsed.get('dr')}")

    model, variable, cfg = load_model_and_variable(best_dir, TRAIN_PDE)

    rows = []
    for test_scale, test_pde in TEST_PDES.items():
        in_dist = (test_scale == TRAIN_TEST_SCALE)
        print(f"  -> {test_scale} ... ", end="", flush=True)

        out_dir = output_dir / "best_run_cross_scale" / f"test_{test_scale}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics, u_true, u_pred = run_eval(model, variable, test_pde, cfg)

            _np.savez_compressed(
                out_dir / "eval.npz",
                u_pred=_np.array(u_pred),
                u_true=_np.array(u_true),
            )
            with (out_dir / "metrics.json").open("w") as f:
                json.dump(
                    {k: (float(v) if v is not None else None) for k, v in metrics.items()},
                    f, indent=2,
                )

            row = {
                "run": best_dir.name,
                **{k: v for k, v in parsed.items() if k != "path"},
                "test_scale": test_scale,
                "in_dist": in_dist,
                "status": "ok",
                **{k: (float(v) if v is not None else None) for k, v in metrics.items()},
            }
            rows.append(row)
            print(f"errr={metrics.get('errr', float('nan')):.4f}" + ("  [in-dist]" if in_dist else ""))

        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({
                "run": best_dir.name, **{k: v for k, v in parsed.items() if k != "path"},
                "test_scale": test_scale, "in_dist": in_dist,
                "status": f"failed: {e}",
            })

    csv_path = output_dir / "best_run_cross_scale_summary.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCross-scale summary -> {csv_path}")

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-augmentation / dropout analysis")
    parser.add_argument("--output-dir", default="eval_results_data_aug")
    parser.add_argument("--skip-plots", action="store_true", help="Skip the training-curve plots.")
    parser.add_argument("--skip-cross-scale", action="store_true", help="Skip cross-scale eval of the best run.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        print(f"Discovering runs in {ROOT_DIR} ...")
        runs = discover_runs(ROOT_DIR)
        print(f"  found {len(runs)} run(s) matching the naming convention")

        best_sigma = plot_sigma_sweep(runs, output_dir)
        plot_dropout_sweep(runs, best_sigma, output_dir)

    if not args.skip_cross_scale:
        evaluate_best_run_cross_scale(ROOT_DIR, output_dir)