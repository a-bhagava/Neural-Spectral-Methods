import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import ast
import jax
import jax.numpy as jnp
from analysis.evaluate_mixed_scale import MODEL_ROOT_DIRS
import numpy as _np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.pde.navierstokes import re4_2, re4_4, re4_8, re4_1, re4_3, re4_6, re4_10
from src import *
from src.pde import *
from src.model import *
from src.basis import *
from src.train import eval as train_eval

# TODO: figure out a script to control for length scale and feed the same IC into these different length scale models

# ── config ───────────────────────────────────────────────────────────────────

NSM_ROOT_DIRS = {
    "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_2"),
    "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_4"),
    "L=0.8": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/nsm/re4_8"),
}

UNET_ROOT_DIRS = {
    "L=0.2": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_2"),
    "L=0.4": Path("/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/2Dvorticity/unet/re4_4"),
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

def plot_layerwise_divergence(results: dict, out_path: Path, title: str):
    """
    results: {test_scale: {layer_name: divergence}}
    Plots divergence vs layer depth for each test scale.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # get layer names from first test scale
    first = next(iter(results.values()))
    layer_names = list(first["divergence"].keys())
    x = np.arange(len(layer_names))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (test_scale, data), color in zip(results.items(), colors):
        divs = [data["divergence"][l] for l in layer_names]
        ax1.plot(x, divs, marker="o", label=test_scale, color=color)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Normalized L2 divergence (↓ = more similar)")
    ax1.set_title("Layer-wise activation divergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def flatten_intermediates(state):
    """
    Unwrap capture_intermediates output:
    {"intermediates": {"layer_name": array_or_tuple}}
    → {"layer_name": array}
    """
    intermediates = state.get("intermediates", state)
    
    flat = {}
    for layer_name, val in intermediates.items():
        # sow wraps in a tuple — unwrap it
        if isinstance(val, tuple):
            flat[layer_name] = val[-1]
        else:
            flat[layer_name] = val
    
    return flat


def collect_intermediates_single_sample(model, variable, pde, cfg):
    """
    Collect intermediate activations for a single sample from pde.solution.
    Returns flattened dict: layer_name -> array
    """
    ϕ, s, u_true = pde.solution

    prng = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs_dict = next(RNGS(prng, ["params", "sample"]))

    ϕ_single = ϕ.map(lambda c: c[0])

    _, state = model.apply(
        variable,
        ϕ_single,
        capture_intermediates=True,
        method=model.forward,
        rngs=rngs_dict,
    )
    flat = flatten_intermediates(state)

    # keep only sow'd layers — filter by known prefixes rather than hardcoded names
    sow_prefixes = (
        "lifting_layer",
        "encoder_level_",
        "bottleneck_",
        "decoder_level_",
        "final_projection_layer",
        "spectralconv_",   # NSM layers
    )
    filtered = {k: v for k, v in flat.items() 
                if any(k.startswith(p) for p in sow_prefixes)}
    
    # sort by natural order so plots are sequential
    def sort_key(name):
        order = ["lifting_layer", "encoder_level_", "bottleneck_", 
                 "decoder_level_", "final_projection_layer", "spectralconv_"]
        for i, prefix in enumerate(order):
            if name.startswith(prefix):
                # extract trailing number if present
                suffix = name[len(prefix):]
                num = int(suffix) if suffix.isdigit() else 0
                return (i, num)
        return (999, 0)
    
    return dict(sorted(filtered.items(), key=lambda x: sort_key(x[0])))


def collect_intermediates_multiple_samples(model, variable, pde, cfg, n_samples=32):
    """
    Collect intermediate activations for multiple samples from pde.solution.
    Returns flattened dict: layer_name -> array [n_samples, ...]
    """
    ϕ, s, u_true = pde.solution

    prng = jax.random.PRNGKey(cfg.get("seed", 0))
    rngs_dict = next(RNGS(prng, ["params", "sample"]))

    ϕ_multiple = ϕ.map(lambda c: c[:n_samples])

    # forward pass for a single sample — same as single sample version
    def single_forward(ϕ_single):
        _, state = model.apply(
            variable,
            ϕ_single,
            capture_intermediates=True,
            method=model.forward,
            rngs=rngs_dict,
        )
        flat = flatten_intermediates(state)
        sow_prefixes = (
            "lifting_layer", "encoder_level_", "bottleneck_",
            "decoder_level_", "final_projection_layer", "spectralconv_",
        )
        return {k: v for k, v in flat.items()
                if any(k.startswith(p) for p in sow_prefixes)}

    # cmap vmaps over n_samples — same pattern as your eval script
    intermediates = utils.cmap(single_forward, cfg.get("vmap"))(ϕ_multiple)

    # sort layers
    def sort_key(name):
        order = ["lifting_layer", "encoder_level_", "bottleneck_",
                 "decoder_level_", "final_projection_layer", "spectralconv_"]
        for i, prefix in enumerate(order):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                num = int(suffix) if suffix.isdigit() else 0
                return (i, num)
        return (999, 0)

    return dict(sorted(intermediates.items(), key=lambda x: sort_key(x[0])))


def cka_similarity(X, Y):
    """
    Linear CKA with proper double-centering of Gram matrices.
    X, Y: [n_samples, n_features]
    """
    def center_gram(K):
        n = K.shape[0]
        H = _np.eye(n) - _np.ones((n, n)) / n
        return H @ K @ H

    # compute Gram matrices
    K_X = X @ X.T  # [n, n]
    K_Y = Y @ Y.T  # [n, n]
    K_X_c = center_gram(K_X)
    K_Y_c = center_gram(K_Y)

    # HSIC
    hsic    = _np.trace(K_X_c @ K_Y_c) / (X.shape[0] - 1) ** 2
    norm_X  = _np.sqrt(_np.trace(K_X_c @ K_X_c) / (X.shape[0] - 1) ** 2)
    norm_Y  = _np.sqrt(_np.trace(K_Y_c @ K_Y_c) / (X.shape[0] - 1) ** 2)

    return float(hsic / (norm_X * norm_Y + 1e-8))


def plot_layerwise_cka_grid(all_results: dict, out_path: Path, title: str = None):
    """
    all_results: {train_scale: {test_scale: {"cka": {layer_name: score}}}}
    3 subplots side by side, one per training scale.
    CKA ranges from 0 (dissimilar) to 1 (identical).
    """
    train_scales = list(all_results.keys())
    n_train = len(train_scales)

    fig, axes = plt.subplots(1, n_train, figsize=(7 * n_train, 5), sharey=True)
    if n_train == 1:
        axes = [axes]

    # consistent colors across subplots
    all_test_scales = list(next(iter(all_results.values())).keys())
    colors = {s: c for s, c in zip(
        all_test_scales,
        plt.cm.viridis(_np.linspace(0, 1, len(all_test_scales)))
    )}

    for ax, train_scale in zip(axes, train_scales):
        results = all_results[train_scale]
        layer_names = list(next(iter(results.values()))["cka"].keys())
        x = _np.arange(len(layer_names))

        for test_scale, data in results.items():
            ckas = [data["cka"][l] for l in layer_names]
            linestyle = "-"  if test_scale == train_scale else "--"
            linewidth = 2.5  if test_scale == train_scale else 1.5
            ax.plot(
                x, ckas,
                marker="o",
                label=test_scale,
                color=colors[test_scale],
                linestyle=linestyle,
                linewidth=linewidth,
            )

        # reference line at CKA=1 (perfect similarity)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

        ax.set_title(f"Train scale: {train_scale}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("CKA similarity (↑ = more similar to training scale)")
    fig.suptitle(
        title or "Layer-wise CKA similarity across length scales",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_layerwise_divergence_grid(all_results: dict, out_path: Path):
    """
    all_results: {train_scale: {test_scale: {layer_name: divergence}}}
    3 subplots side by side, one per training scale.
    """
    train_scales = list(all_results.keys())
    fig, axes = plt.subplots(1, len(train_scales), figsize=(7 * len(train_scales), 5), sharey=True)
    
    if len(train_scales) == 1:
        axes = [axes]

    # consistent colors across subplots
    all_test_scales = list(next(iter(all_results.values())).keys())
    colors = {s: c for s, c in zip(all_test_scales, plt.cm.viridis(_np.linspace(0, 1, len(all_test_scales))))}

    for ax, train_scale in zip(axes, train_scales):
        results = all_results[train_scale]
        layer_names = list(next(iter(results.values()))["divergence"].keys())
        x = _np.arange(len(layer_names))

        for test_scale, data in results.items():
            divs = [data["divergence"][l] for l in layer_names]
            linestyle = "-" if test_scale == train_scale else "--"
            linewidth = 2.5 if test_scale == train_scale else 1.5
            ax.plot(x, divs, marker="o", label=test_scale,
                    color=colors[test_scale],
                    linestyle=linestyle, linewidth=linewidth)

        ax.set_title(f"Train scale: {train_scale}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalized L2 divergence (↓ = more similar)")
    fig.suptitle("UNET: layer-wise activation divergence across length scales", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main(divergence=True, pca=False, cka=False):
    all_scales = {
        "L=0.1": OUT_OF_DIST_PDES["L=0.1"],
        "L=0.2": IN_DIST_PDES["L=0.2"],
        "L=0.3": OUT_OF_DIST_PDES["L=0.3"],
        "L=0.4": IN_DIST_PDES["L=0.4"],
        "L=0.6": OUT_OF_DIST_PDES["L=0.6"],
        "L=0.8": IN_DIST_PDES["L=0.8"],
        "L=1.0": OUT_OF_DIST_PDES["L=1.0"],
    }

    train_scale_dirs = {
        "L=0.2": UNET_ROOT_DIRS["L=0.2"],
        "L=0.4": UNET_ROOT_DIRS["L=0.4"],
        "L=0.8": UNET_ROOT_DIRS["L=0.8"],
    }

    if divergence: 
        output_dir = Path("analysis/single_scale_representations")
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = {}

        for train_scale, root_dir in train_scale_dirs.items():
            train_pde = IN_DIST_PDES[train_scale]
            run_dir = find_seed_runs(root_dir)[0]
            model, variable, cfg = load_model_and_variable(run_dir, train_pde)
            print(f"\nLoaded {train_scale}: {run_dir.name}")

            # collect activations at all scales using this model
            all_acts = {}
            for scale, pde in all_scales.items():
                print(f"  collecting activations at {scale}...")
                all_acts[scale] = collect_intermediates_single_sample(model, variable, pde, cfg)

            # compute divergence vs this model's training scale
            acts_train = all_acts[train_scale]
            layer_names = list(acts_train.keys())
            results = {}

            for test_scale, acts_test in all_acts.items():
                layer_divs = {}
                for layer_name in layer_names:
                    z_train = _np.array(acts_train[layer_name]).reshape(-1)
                    z_test  = _np.array(acts_test[layer_name]).reshape(-1)
                    div = _np.linalg.norm(z_train - z_test) / (_np.linalg.norm(z_train) + 1e-8)
                    layer_divs[layer_name] = float(div)

                results[test_scale] = {"divergence": layer_divs}
                avg_div = _np.mean(list(layer_divs.values()))
                print(f"    {test_scale}: avg divergence = {avg_div:.4f}")

            all_results[train_scale] = results

        # plot all three side by side
        plot_layerwise_divergence_grid(
            all_results,
            out_path=output_dir / "unet_all_train_scales_layerwise_divergence.png",
        )

        # save raw results
        with (output_dir / "unet_all_train_scales_divergence.json").open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_dir}")

    if cka: 
        output_dir = Path("analysis/single_scale_representations")
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = {}

        for train_scale, root_dir in train_scale_dirs.items():
            train_pde = IN_DIST_PDES[train_scale]
            run_dir = find_seed_runs(root_dir)[0]
            model, variable, cfg = load_model_and_variable(run_dir, train_pde)
            print(f"\nLoaded {train_scale}: {run_dir.name}")

            # collect activations at all scales using this model
            all_acts = {}
            for scale, pde in all_scales.items():
                print(f"  collecting activations at {scale}...")
                all_acts[scale] = collect_intermediates_multiple_samples(model, variable, pde, cfg, n_samples=128)
            
            # compute CKA similarity vs this model's training scale
            acts_train = all_acts[train_scale]
            layer_names = list(acts_train.keys())
            results = {}
            for test_scale, acts_test in all_acts.items():
                layer_ckas = {}
                for layer_name in layer_names:
                    z_train = _np.array(acts_train[layer_name]).reshape(128, -1)
                    z_test  = _np.array(acts_test[layer_name]).reshape(128, -1)
                    cka_score = cka_similarity(z_train, z_test)
                    layer_ckas[layer_name] = float(cka_score)

                results[test_scale] = {"cka": layer_ckas}
                avg_cka = _np.mean(list(layer_ckas.values()))
                print(f"    {test_scale}: avg CKA similarity = {avg_cka:.4f}")

            all_results[train_scale] = results
        
        # plotting
        plot_layerwise_cka_grid(all_results, out_path=output_dir / "nsm_layerwise_cka_grid.png", 
                                title="NSM: layer-wise CKA similarity vs training scale")

        with (output_dir / "nsm_layerwise_cka.json").open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_dir}")
    
    if pca:
        output_dir = Path("analysis/single_scale_representations/pca")
        output_dir.mkdir(parents=True, exist_ok=True)

        pca_layers_NSM = ["spectralconv_4", "final_projection_layer"]
        pca_layers_UNET = ["encoder_level_0", "bottleneck_2", "final_projection_layer"]

        train_scale_dirs = {
            "L=0.2": UNET_ROOT_DIRS["L=0.2"],
            "L=0.4": UNET_ROOT_DIRS["L=0.4"],
            "L=0.8": UNET_ROOT_DIRS["L=0.8"],
        }

        scale_colors = {
            "L=0.1": "#d62728",
            "L=0.2": "#ff7f0e",
            "L=0.3": "#bcbd22",
            "L=0.4": "#2ca02c",
            "L=0.6": "#17becf",
            "L=0.8": "#1f77b4",
            "L=1.0": "#9467bd",
        }

        n_samples = 128
        n_layers_per_model = 3
        n_train_scales = len(train_scale_dirs)

        fig, axes = plt.subplots(
            n_layers_per_model, n_train_scales,
            figsize=(6 * n_train_scales, 6 * n_layers_per_model),
        )
        if n_layers_per_model == 1:
            axes = axes[_np.newaxis, :]
        if n_train_scales == 1:
            axes = axes[:, _np.newaxis]

        for col_idx, (train_scale, root_dir) in enumerate(train_scale_dirs.items()):
            train_pde = IN_DIST_PDES[train_scale]
            run_dir = find_seed_runs(root_dir)[0]
            model, variable, cfg = load_model_and_variable(run_dir, train_pde)
            pca_layers = pca_layers_NSM if "nsm" in str(root_dir) else pca_layers_UNET

            print(f"\nLoaded {train_scale}: {run_dir.name}")
            all_scale_acts_raw = {}
            for input_scale, input_pde in all_scales.items():
                print(f"  collecting {input_scale}...")
                all_scale_acts_raw[input_scale] = collect_intermediates_multiple_samples(
                    model, variable, input_pde, cfg, n_samples=n_samples
                )

            for row_idx, layer_name in enumerate(pca_layers):
                ax = axes[row_idx, col_idx]

                # extract and flatten this layer for all scales
                scale_acts = {}
                for input_scale, acts in all_scale_acts_raw.items():
                    scale_acts[input_scale] = _np.array(
                        acts[layer_name]
                    ).reshape(n_samples, -1)

                # fit scaler + PCA on training scale only
                scaler = StandardScaler()
                Z_train_scaled = scaler.fit_transform(scale_acts[train_scale])
                pca_model = PCA(n_components=2)
                pca_model.fit(Z_train_scaled)

                # compute clip bounds from training scale PCA distribution
                z_train_pca = pca_model.transform(Z_train_scaled)
                clip_factor = 3.0
                clip_lo = clip_factor * z_train_pca.std(axis=0) * -1
                clip_hi = clip_factor * z_train_pca.std(axis=0)

                # project all scales, clip outliers, scatter
                for input_scale, z in scale_acts.items():
                    z_scaled = scaler.transform(z)
                    z_pca = pca_model.transform(z_scaled)
                    z_pca = _np.clip(z_pca, clip_lo, clip_hi)
                    ax.scatter(
                        z_pca[:, 0], z_pca[:, 1],
                        color=scale_colors[input_scale],
                        label=input_scale,
                        alpha=0.7,
                        s=15,
                    )

                var_explained = pca_model.explained_variance_ratio_
                ax.set_xlabel(f"PC 1 ({var_explained[0]*100:.1f}%)")
                ax.set_ylabel(f"PC 2 ({var_explained[1]*100:.1f}%)")
                ax.set_title(f"Train {train_scale} | {layer_name}", fontsize=10)
                ax.set_xscale("linear")
                ax.set_yscale("linear")
                ax.grid(True, alpha=0.2)

                if col_idx == 0:
                    ax.legend(fontsize=7, markerscale=2)

        fig.suptitle("UNET: PCA of activations (scaler + PCA fit on training scale only)", fontsize=12)
        fig.tight_layout()
        out_path = output_dir / "unet_pca_all_scales_grid.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_path}")
        
if __name__ == "__main__":
    main(pca=False, divergence=False, cka=True)