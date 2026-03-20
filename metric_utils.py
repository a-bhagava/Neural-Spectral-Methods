import os
from pathlib import Path
import numpy as np

def hyperparameter_results(root_dir, log_scale=True, plotting=True):
    residual_errors = {}
    absolute_errors = {}
    subdir, dirs, files = next(os.walk(root_dir))

    # collect the residual and absolute errors for each run
    for run in dirs:
        run_dir = os.path.join(root_dir, run)
        residual_errors[run] = os.path.join(run_dir, "metric.residual.npy")
        absolute_errors[run] = os.path.join(run_dir, "metric.erra.npy")

    if plotting: 
        fig, (ax_res, ax_abs) = plt.subplots(1, 2, figsize=(16, 6))

        for key in residual_errors:
            res_path = residual_errors[key]
            abs_path = absolute_errors[key]

            # run format: 'NSM_hdim16_depth5_mode12_31_31_lr0.0005'
            parts = key.split("_")
            hdim  = parts[1]   # 'hdim16'
            depth = parts[2]   # 'depth5'
            mode  = "_".join(parts[3:6])   # 'mode12_31_31'
            lr    = parts[6]   # 'lr0.0005'
            label = f"{hdim}, {depth}, {mode}, {lr}"

            if Path(res_path).is_file():
                data = np.load(res_path)
                ax_res.plot(data, label=label)

            if Path(abs_path).is_file():
                data = np.load(abs_path)
                ax_abs.plot(data, label=label)

        for ax, title, ylabel in [
            (ax_res, "PDE Residual vs Training Iterations",   "PDE Residual"),
            (ax_abs, "Absolute Error vs Training Iterations", "Absolute Error"),
        ]:
            ax.set_xlabel("Training Iterations")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if log_scale:
                ax.set_yscale("log")

        plt.tight_layout()
        plt.show()

    return residual_errors, absolute_errors

def find_best_run(root_dir):
    _, dirs, _ = next(os.walk(root_dir))
    final_values = {}

    for run in dirs:
        run_dir = os.path.join(root_dir, run)
        res_path = os.path.join(run_dir, "metric.residual.npy")
        abs_path = os.path.join(run_dir, "metric.erra.npy")

        if not Path(res_path).is_file() or not Path(abs_path).is_file():
            continue

        final_values[run] = {
            "residual": np.load(res_path)[-1],
            "absolute": np.load(abs_path)[-1],
        }

    if not final_values:
        raise ValueError("No complete runs found in root_dir")

    # rank each run per metric (1 = best), then average the ranks
    for metric in ("residual", "absolute"):
        sorted_runs = sorted(final_values, key=lambda r: final_values[r][metric])
        for rank, run in enumerate(sorted_runs, start=1):
            final_values[run].setdefault("rank_sum", 0)
            final_values[run]["rank_sum"] += rank

    best_run = min(final_values, key=lambda r: final_values[r]["rank_sum"])

    print(f"Best run: {best_run}")
    print(f"  Final residual error : {final_values[best_run]['residual']:.6f}")
    print(f"  Final absolute error : {final_values[best_run]['absolute']:.6f}")
    print(f"  Average rank         : {final_values[best_run]['rank_sum'] / 2:.1f}")

    return best_run, final_values