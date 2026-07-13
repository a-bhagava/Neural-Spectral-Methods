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


# def collect_intermediates(model, variable, pde, cfg):
#     """
#     Collect intermediate activations for all samples in pde.solution
#     using the same vmap pattern as your eval script.
#     Returns dict: layer_name -> array [N, ...]
#     """
#     ϕ, s, u_true = pde.solution

#     prng = jax.random.PRNGKey(cfg.get("seed", 0))
#     rngs = RNGS(prng, ["params", "sample"])

#     def single_forward(ϕ_single):
#         _, state = model.apply(
#             variable,
#             ϕ_single,
#             capture_intermediates=True,
#             mutable=["intermediates"],
#             method=model.forward,
#             rngs=dict(rngs),
#         )
#         return state["intermediates"]

#     # cmap vmaps over the batch dim — same as your eval
#     # do it for a single input sample and also save image of the sample too
#     intermediates = utils.cmap(single_forward, cfg.get("vmap"))(ϕ)
#     return intermediates


# def flatten_intermediates(intermediates):
#     """
#     capture_intermediates returns:
#     {layer_name: {'__call__': (array,)}}
#     Flatten to: {layer_name: array}
#     """
#     flat = {}
#     for layer_name, layer_dict in intermediates.items():
#         if isinstance(layer_dict, dict) and '__call__' in layer_dict:
#             val = layer_dict['__call__']
#             # sow returns a tuple — take the last element
#             if isinstance(val, tuple):
#                 flat[layer_name] = val[-1]
#             else:
#                 flat[layer_name] = val
#         else:
#             flat[layer_name] = layer_dict
#     return flat

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


# def collect_intermediates_single_sample(model, variable, pde, cfg):
#     """
#     Collect intermediate activations for a single sample from pde.solution.
#     Returns flattened dict: layer_name -> array [cheb, m1, m2, hdim]
#     """
#     ϕ, s, u_true = pde.solution

#     prng = jax.random.PRNGKey(cfg.get("seed", 0))
#     rngs = RNGS(prng, ["params", "sample"])
#     rngs_dict = next(rngs)

#     # take just the 0th sample
#     ϕ_single = ϕ.map(lambda c: c[0])

#     _, state = model.apply(
#         variable,
#         ϕ_single,
#         capture_intermediates=True,
#         method=model.forward,
#         rngs=rngs_dict,
#     )
#     flat = flatten_intermediates(state)
        
#     # keep only sow'd layers
#     sow_layers = ["lifting_layer"] + \
#                 [f"spectralconv_{i}" for i in range(10)] + \
#                 ["final_projection_layer"]
#     return {k: v for k, v in flat.items() if k in sow_layers}


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


def main():
    output_dir = Path("analysis/single_scale_representations")
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()

# def main():
#     output_dir = Path("analysis/single_scale_representations")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     all_scales = {
#         "L=0.1": OUT_OF_DIST_PDES["L=0.1"],
#         "L=0.2": IN_DIST_PDES["L=0.2"],
#         "L=0.3": OUT_OF_DIST_PDES["L=0.3"],
#         "L=0.4": IN_DIST_PDES["L=0.4"],
#         "L=0.6": OUT_OF_DIST_PDES["L=0.6"],
#         "L=0.8": IN_DIST_PDES["L=0.8"],
#         "L=1.0": OUT_OF_DIST_PDES["L=1.0"],
#     }

#     # load L=0.8 model — single seed
#     run_dir = find_seed_runs(NSM_ROOT_DIRS["L=0.8"])[0]
#     train_pde = IN_DIST_PDES["L=0.8"]
#     model, variable, cfg = load_model_and_variable(run_dir, train_pde)
#     print(f"Loaded: {run_dir.name}")

#     # collect single-sample activations at every scale
#     all_acts = {}
#     for scale, pde in all_scales.items():
#         print(f"Collecting activations at {scale}...")
#         all_acts[scale] = collect_intermediates_single_sample(model, variable, pde, cfg)

#     layer_names = list(all_acts["L=0.8"].keys())
#     print("Layer names:", layer_names)
#     print("Sample shape:", _np.array(all_acts["L=0.8"][layer_names[0]]).shape)

#     # compute layer-wise L2 divergence vs L=0.8 (training scale)
#     print("\nComputing layer-wise divergence vs L=0.8...")
#     acts_train = all_acts["L=0.8"]
#     layer_names = list(acts_train.keys())

#     results = {}
#     for test_scale, acts_test in all_acts.items():
#         layer_divs = {}
#         for layer_name in layer_names:
#             z_train = _np.array(acts_train[layer_name]).reshape(-1)  # flatten to 1D
#             z_test  = _np.array(acts_test[layer_name]).reshape(-1)

#             div = _np.linalg.norm(z_train - z_test) / (_np.linalg.norm(z_train) + 1e-8)
#             layer_divs[layer_name] = float(div)

#         results[test_scale] = {"divergence": layer_divs}
#         avg_div = _np.mean(list(layer_divs.values()))
#         print(f"{test_scale}: avg divergence = {avg_div:.4f}")

#     # plot
#     plot_layerwise_divergence(
#         results,
#         out_path=output_dir / "nsm_L08_layerwise_divergence_single_sample.png",
#         title="NSM (trained L=0.8): layer-wise L2 divergence vs training scale\n(single sample per scale)"
#     )

#     # save raw results
#     with (output_dir / "nsm_L08_divergence_single_sample.json").open("w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nResults saved to {output_dir}")
