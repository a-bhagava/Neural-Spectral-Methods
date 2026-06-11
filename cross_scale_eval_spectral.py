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

SCALES = list(ROOT_DIRS.keys())

# ── spectral helpers ──────────────────────────────────────────────────────────

Field = List[Array]
Shape = Tuple[int, ...]

def shape(xs: Field) -> Shape:
    ns, = set(map(np.shape, xs))
    return ns

def fft(xs: Field) -> Field:
    assert len(set(map(np.shape, xs))) == 1
    return [np.fft.fftn(x, norm="forward") for x in xs]

def ifft(xs_: Field) -> Field:
    assert len(set(map(np.shape, xs_))) == 1
    return [np.fft.irfftn(x_, x_.shape, norm="forward") for x_ in xs_]

def laplacian(*ns: int) -> Array:
    ks = wavenumber(*ns, complex=False)
    return -sum(k ** 2 for k in ks).astype(complex)

def wavenumber(*ns: int, complex=True):
    freq = np.array(1.0)
    if complex: freq *= 1j
    k = lambda n: np.fft.fftfreq(n).astype(freq.dtype) * freq * n
    return np.meshgrid(*map(k, ns), indexing="ij")

def curl(xs: Field, fourier=False) -> Field:
    def cross(xs, ys):
        return [xs[1]*ys[2] - xs[2]*ys[1],
                xs[2]*ys[0] - xs[0]*ys[2],
                xs[0]*ys[1] - xs[1]*ys[0]]
    if fourier: xs_ = xs
    else: xs_ = fft(xs)
    ks = wavenumber(*shape(xs))
    xs_ = cross(ks, xs_)
    if fourier: return xs_
    else: return ifft(xs_)

def vor2vel(ws: Field, fourier=False) -> Field:
    if len(ws) == 1:
        assert len(shape(ws)) == 2
        wz  = ws[0][:, :, None]
        wxy = [np.zeros(wz.shape)] * 2
        vs  = vor2vel(wxy + [wz], fourier)
        return [v.squeeze(-1) for v in vs[:2]]
    if len(ws) == 3:
        assert len(shape(ws)) == 3
        if fourier: ws_ = ws
        else: ws_ = fft(ws)
        lap = laplacian(*shape(ws)).at[0, 0, 0].set(1.0)
        vs_ = [-w_ / lap for w_ in curl(ws_, True)]
        if fourier: return vs_
        else: return ifft(vs_)
    raise ValueError(f"invalid dimension {len(ws)}")

def energy_spectrum(vs: Field) -> Array:
    vs_ = fft(vs)
    v2  = sum(v_ * v_.conj() for v_ in vs_).real
    ks  = wavenumber(*shape(vs), complex=False)
    k   = np.round(np.sqrt(sum(k**2 for k in ks))).astype(int)
    Ek  = np.zeros(np.max(k) + 1)
    return Ek.at[k].add(v2)

def compute_Ek_trajectory(data):
    """
    Args:
        data: (batch, time, nx, ny, 1) vorticity
    Returns:
        Ek_t:    (time, nk) ensemble mean  E(k)
        Ek_t_sd: (time, nk) ensemble std   E(k)
    """
    batch, time, nx, ny, _ = data.shape
    Ek_t    = None
    Ek_t_sd = None

    for t in range(time):
        Ek_batch = []
        for b in range(batch):
            omega = np.array(data[b, t, :, :, 0])
            vs    = vor2vel([omega])
            Ek    = _np.array(energy_spectrum(vs))
            Ek_batch.append(Ek)

        Ek_arr = _np.stack(Ek_batch, axis=0)          # (batch, nk)

        if Ek_t is None:
            nk      = Ek_arr.shape[1]
            Ek_t    = _np.zeros((time, nk))
            Ek_t_sd = _np.zeros((time, nk))

        Ek_t[t]    = Ek_arr.mean(axis=0)
        Ek_t_sd[t] = Ek_arr.std(axis=0)

        if t % 8 == 0:
            print(f"      t={t}/{time}", flush=True)

    return Ek_t, Ek_t_sd

def ralsd(Ek_model, Ek_gt, k_min=1, k_max=None):
    """
    Args:
        Ek_model: (time, nk) ensemble mean E(k)
        Ek_gt:    (time, nk) ensemble mean E(k)
    Returns:
        (time,) RALSD at each timestep
    """
    if k_max is None:
        k_max = Ek_model.shape[-1]
    m = _np.maximum(Ek_model[:, k_min:k_max], 1e-10)
    g = _np.maximum(Ek_gt[:,    k_min:k_max], 1e-10)
    return _np.sqrt(_np.mean((_np.log(m) - _np.log(g))**2, axis=-1))  # (time,)

# ── reuse from cross_scale_eval.py ───────────────────────────────────────────

def find_best_run(root_dir: Path) -> tuple[Path, dict]:
    best_path, best_metrics, best_residual = None, None, float("inf")
    for run_dir in sorted(root_dir.iterdir()):
        if not run_dir.is_dir(): continue
        ckpt_path     = run_dir / "variable_ckpt.npy"
        residual_path = run_dir / "metric.residual.npy"
        if not ckpt_path.exists() or not residual_path.exists(): continue
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
            print(f"  Warning: could not read {run_dir.name}: {e}")
    if best_path is None:
        raise RuntimeError(f"No valid runs found in {root_dir}")
    print(f"  Best run : {best_path.name}")
    print(f"  Residual : {best_residual:.6f}")
    return best_path, best_metrics

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
    from src.train import eval as train_eval
    train = Trainer(mod=model, pde=test_pde, cfg=cfg)
    prng  = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs  = RNGS(prng, ["params", "sample"])
    metrics, predictions = train.apply(
        {}, variable, method=train_eval, rngs=next(rngs),
    )
    u_true, u_pred = predictions
    return metrics, u_true, u_pred

def load_or_run_eval(model, variable, test_pde, cfg, pair_dir: Path):
    """Load predictions from eval.npz if it exists, otherwise run eval."""
    eval_path = pair_dir / "eval.npz"
    if eval_path.exists():
        print(f"    Loading cached predictions from {eval_path}", flush=True)
        data = _np.load(eval_path, allow_pickle=True)
        # adapt these keys to whatever your eval.npz actually contains
        u_true = data["u_true"]
        u_pred = data["u_pred"]
        # reconstruct a metrics dict from whatever is saved, or return empty
        metrics = {k: data[k] for k in data.files if k not in ("u_true", "u_pred")}
        return metrics, u_true, u_pred
    else:
        print(f"    No cached eval found, running model...", flush=True)
        metrics, u_true, u_pred = run_eval(model, variable, test_pde, cfg)
        # optionally save for next time
        pair_dir.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(eval_path, u_true=u_true, u_pred=u_pred, **metrics)
        return metrics, u_true, u_pred

# ── spectral plotting ─────────────────────────────────────────────────────────

def plot_spectral_pair(Ek_true, Ek_pred, Ek_true_sd, Ek_pred_sd,
                       ralsd_curve, out_path: Path, title: str):
    """
    Three-panel figure for one (train_scale, test_scale) pair:
        Left:   time-averaged E(k)
        Middle: final snapshot E(k)
        Right:  RALSD(t)
    """
    nk = Ek_true.shape[1]
    ks = _np.arange(nk)
    T  = Ek_true.shape[0]
    ts = _np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- panel 1: time-averaged E(k) ---
    ax = axes[0]
    for Ek_t, label, color, ls, lw in [
        (Ek_true, 'Ground Truth', '#2C2C2A', '-',  1.8),
        (Ek_pred, 'Prediction',   '#378ADD', '--', 1.4),
    ]:
        Ek_mean = Ek_t.mean(axis=0)
        ax.loglog(ks[1:], Ek_mean[1:], color=color, linestyle=ls,
                  linewidth=lw, label=label)

    mid = nk // 3
    E0  = Ek_true.mean(axis=0)[mid]
    k_r = _np.array([ks[max(1, nk//8)], ks[nk*3//4]])
    ax.loglog(k_r, E0 * (k_r / ks[mid])**(-5/3),
              'k:', linewidth=1.0, alpha=0.5, label=r'$k^{-5/3}$')
    ax.set_xlabel(r'$k$');  ax.set_ylabel(r'$E(k)$')
    ax.set_title('Time-averaged $E(k)$', fontsize=11)
    ax.legend(fontsize=8);  ax.grid(True, which='both', alpha=0.2)

    # --- panel 2: final snapshot E(k) ---
    ax = axes[1]
    for Ek_t, label, color, ls, lw in [
        (Ek_true, 'Ground Truth', '#2C2C2A', '-',  1.8),
        (Ek_pred, 'Prediction',   '#378ADD', '--', 1.4),
    ]:
        ax.loglog(ks[1:], Ek_t[-1, 1:], color=color, linestyle=ls,
                  linewidth=lw, label=label)

    E0_final = Ek_true[-1, mid]
    ax.loglog(k_r, E0_final * (k_r / ks[mid])**(-5/3),
              'k:', linewidth=1.0, alpha=0.5, label=r'$k^{-5/3}$')
    ax.set_xlabel(r'$k$');  ax.set_ylabel(r'$E_T(k)$')
    ax.set_title('Final Snapshot $E_T(k)$', fontsize=11)
    ax.legend(fontsize=8);  ax.grid(True, which='both', alpha=0.2)

    # --- panel 3: RALSD(t) ---
    ax = axes[2]
    ax.plot(ts, ralsd_curve, color='#378ADD', linewidth=1.4)
    ax.set_xlabel('Timestep $t$');  ax.set_ylabel('RALSD')
    ax.set_title('RALSD$(t)$', fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cross_scale_summary(ralsd_table: dict, output_dir: Path):
    """
    3x3 heatmap of time-mean RALSD, train scale on rows, test scale on cols.
    Diagonal = in-distribution.
    """
    n      = len(SCALES)
    matrix = _np.full((n, n), _np.nan)

    for i, train in enumerate(SCALES):
        for j, test in enumerate(SCALES):
            val = ralsd_table.get((train, test))
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    fig.colorbar(im, ax=ax, label='Time-mean RALSD')

    ax.set_xticks(range(n)); ax.set_xticklabels(SCALES)
    ax.set_yticks(range(n)); ax.set_yticklabels(SCALES)
    ax.set_xlabel('Test scale');  ax.set_ylabel('Train scale')
    ax.set_title('Cross-scale RALSD (lower = better)', fontsize=12)

    # annotate cells
    for i in range(n):
        for j in range(n):
            if not _np.isnan(matrix[i, j]):
                marker = '*' if i == j else ''
                ax.text(j, i, f"{matrix[i,j]:.3f}{marker}",
                        ha='center', va='center', fontsize=10,
                        color='black')

    fig.tight_layout()
    fig.savefig(output_dir / 'cross_scale_ralsd_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap → {output_dir / 'cross_scale_ralsd_heatmap.png'}")


# ── main loop ─────────────────────────────────────────────────────────────────

def evaluate_all_spectral(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows        = []
    ralsd_table = {}   # (train_scale, test_scale) -> time-mean RALSD scalar

    # Step 1: find best run per training scale
    best_runs = {}
    for scale, root_dir in ROOT_DIRS.items():
        print(f"\nFinding best run for train scale {scale}:")
        run_dir, _ = find_best_run(root_dir)
        best_runs[scale] = run_dir

    # Step 2: load each model once, evaluate across all test scales
    for train_scale, run_dir in best_runs.items():
        train_pde = PDES[train_scale]
        print(f"\nLoading model trained on {train_scale}: {run_dir.name}")
        bound_model, variable, cfg = load_model_and_variable(run_dir, train_pde)

        for test_scale, test_pde in PDES.items():
            ood = train_scale != test_scale
            print(f"\n  → Evaluating train={train_scale} / test={test_scale}"
                  + ("  [OOD]" if ood else "  [in-dist]"), flush=True)
            try:
                # metrics, u_true, u_pred = run_eval(bound_model, variable, test_pde, cfg)
                pair_dir = output_dir / f"train_{train_scale}" / f"test_{test_scale}"
                metrics, u_true, u_pred = load_or_run_eval(bound_model, variable, test_pde, cfg, pair_dir)

                # u_true, u_pred: (batch, time, nx, ny, 1)
                print(f"    Computing spectral metrics...", flush=True)
                Ek_true, Ek_true_sd = compute_Ek_trajectory(_np.array(u_true))
                Ek_pred, Ek_pred_sd = compute_Ek_trajectory(_np.array(u_pred))

                # RALSD per timestep, then scalar summary
                ralsd_curve       = ralsd(Ek_pred, Ek_true)        # (time,)
                ralsd_mean_scalar = float(ralsd_curve.mean())       # scalar for CSV
                ralsd_final       = float(ralsd_curve[-1])

                print(f"    RALSD (time-mean): {ralsd_mean_scalar:.4f}")
                print(f"    RALSD (final t):   {ralsd_final:.4f}")

                ralsd_table[(train_scale, test_scale)] = ralsd_mean_scalar

                # save outputs
                pair_dir = output_dir / f"train_{train_scale}" / f"test_{test_scale}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                _np.savez_compressed(pair_dir / "spectra.npz",
                                     Ek_true=Ek_true, Ek_pred=Ek_pred,
                                     Ek_true_sd=Ek_true_sd, Ek_pred_sd=Ek_pred_sd,
                                     ralsd_curve=ralsd_curve)

                with (pair_dir / "spectral_metrics.json").open("w") as f:
                    json.dump({
                        "ralsd_mean": ralsd_mean_scalar,
                        "ralsd_final": ralsd_final,
                        "train_scale": train_scale,
                        "test_scale":  test_scale,
                        "ood": ood,
                    }, f, indent=2)

                plot_spectral_pair(
                    Ek_true, Ek_pred, Ek_true_sd, Ek_pred_sd,
                    ralsd_curve,
                    out_path=pair_dir / "spectral_comparison.png",
                    title=(f"Train {train_scale} → Test {test_scale}"
                           f"  |  RALSD(mean)={ralsd_mean_scalar:.4f}"
                           + ("  [OOD]" if ood else "")),
                )

                rows.append({
                    "model":        run_dir.name,
                    "train_scale":  train_scale,
                    "test_scale":   test_scale,
                    "ood":          ood,
                    "ralsd_mean":   ralsd_mean_scalar,
                    "ralsd_final":  ralsd_final,
                    **{k: (float(v) if v is not None else None)
                       for k, v in metrics.items()},
                })

            except Exception as e:
                import traceback
                print(f"    FAILED: {e}")
                traceback.print_exc()
                rows.append({
                    "model":       run_dir.name,
                    "train_scale": train_scale,
                    "test_scale":  test_scale,
                    "ood":         ood,
                    "status":      f"failed: {e}",
                })

    # save summary CSV
    if rows:
        summary_path = output_dir / "cross_scale_spectral_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV → {summary_path}")

    # 3x3 heatmap
    plot_cross_scale_summary(ralsd_table, output_dir)

    return rows


def print_table(rows: list, metric: str = "ralsd_mean"):
    lookup = {}
    for r in rows:
        if metric in r:
            lookup[(r["train_scale"], r["test_scale"])] = r[metric]

    col_w = 12
    header = f"{'train / test':<14}" + "".join(f"{s:>{col_w}}" for s in SCALES)
    print(f"\n{metric} — cross-scale table (diagonal = in-distribution):")
    print(header)
    print("─" * len(header))
    for train in SCALES:
        row = f"{train:<14}"
        for test in SCALES:
            val    = lookup.get((train, test))
            marker = "*" if train == test else ""
            row   += f"{f'{val:.4f}{marker}':>{col_w}}" if val is not None else f"{'—':>{col_w}}"
        print(row)
    print("  * = in-distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-scale spectral evaluation")
    parser.add_argument("--output-dir", default="spectral_eval_results")
    args = parser.parse_args()

    rows = evaluate_all_spectral(Path(args.output_dir))
    print_table(rows, "ralsd_mean")