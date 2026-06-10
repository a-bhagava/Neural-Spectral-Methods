from src import *
from src.pde import *
from src.model import *
from src.basis import *

Field = List[Array]
Shape = Tuple[int, ...]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as _np

GT_PATH   = "/home/abhagava/orcd/scratch/Neural-Spectral-Methods/src/pde/navierstokes/u.Re=2000:T=3:None.npy"
NSM_PATH  = "/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/nsm_hyperparam/nsT3re4.length0.2/nsm_hdim32_depth15_mode12_41_41_lr0.001/uhat.npy"
UNET_PATH = "/home/abhagava/orcd/scratch/Neural-Spectral-Methods/log/multiscale/unet_hyperparam/nsT3re4.length0.2/hdim32_mode12_64_64_levels4_bottleneck3_lr0.001/uhat.npy"
OUT_PATH  = "/home/abhagava/orcd/scratch/Neural-Spectral-Methods/spectra_comparison.png"


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
    def cross(xs: Field, ys: Field) -> Field:
        return [xs[1] * ys[2] - xs[2] * ys[1],
                xs[2] * ys[0] - xs[0] * ys[2],
                xs[0] * ys[1] - xs[1] * ys[0]]
    if fourier: xs_ = xs
    else: xs_ = fft(xs)
    ns = shape(xs)
    ks = wavenumber(*ns)
    xs_ = cross(ks, xs_)
    if fourier: return xs_
    else: return ifft(xs_)

def vor2vel(ws: Field, fourier=False) -> Field:
    if len(ws) == 1:
        assert len(shape(ws)) == 2
        wz = ws[0][:, :, None]
        wxy = [np.zeros(wz.shape)] * 2
        vs = vor2vel(wxy + [wz], fourier)
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
    k   = np.round(np.sqrt(sum(k ** 2 for k in ks))).astype(int)
    Ek  = np.zeros(np.max(k) + 1)
    return Ek.at[k].add(v2)


# ---------------------------------------------------------------------- #
#                         COMPUTE TRAJECTORIES                           #
# ---------------------------------------------------------------------- #

def compute_Ek_trajectory(data):
    """
    Args:
        data: (batch, time, nx, ny, 1) vorticity
    Returns:
        Ek_t:    (time, nk) ensemble mean  E(k) at each timestep
        Ek_t_sd: (time, nk) ensemble std   E(k) at each timestep
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

        Ek_arr = _np.stack(Ek_batch, axis=0)        # (batch, nk)

        if Ek_t is None:
            nk      = Ek_arr.shape[1]
            Ek_t    = _np.zeros((time, nk))
            Ek_t_sd = _np.zeros((time, nk))

        Ek_t[t]    = Ek_arr.mean(axis=0)
        Ek_t_sd[t] = Ek_arr.std(axis=0)

        if t % 8 == 0:
            print(f"  t={t}/{time}", flush=True)

    return Ek_t, Ek_t_sd                            # (time, nk), (time, nk)


def ralsd(Ek_model, Ek_gt, k_min=1, k_max=None):
    """
    Radially Averaged Log Spectral Distance per timestep.

    Args:
        Ek_model: (time, nk) ensemble mean E(k)
        Ek_gt:    (time, nk) ensemble mean E(k)
    Returns:
        (time,) scalar distance at each timestep
    """
    if k_max is None:
        k_max = Ek_model.shape[-1] // 2
    m = _np.maximum(Ek_model[:, k_min:k_max], 1e-10)
    g = _np.maximum(Ek_gt[:,    k_min:k_max], 1e-10)
    return _np.sqrt(_np.mean((_np.log(m) - _np.log(g)) ** 2, axis=-1))  # (time,)


# ---------------------------------------------------------------------- #
#                              PLOTTING                                  #
# ---------------------------------------------------------------------- #

def plot_spectra_comparison():
    print("Loading data...", flush=True)
    u_gt   = np.load(GT_PATH,   mmap_mode='r')
    u_nsm  = np.load(NSM_PATH,  mmap_mode='r')
    u_unet = np.load(UNET_PATH, mmap_mode='r')
    print(f"Shapes — GT: {u_gt.shape}, NSM: {u_nsm.shape}, UNet: {u_unet.shape}", flush=True)

    datasets = [
        ('Ground Truth', u_gt,   '#2C2C2A', '-',  1.8),
        ('NSM',          u_nsm,  '#378ADD', '--', 1.4),
        ('UNet',         u_unet, '#D85A30', '--', 1.4),
    ]

    # compute E(k) for all datasets
    all_Ek    = {}
    all_Ek_sd = {}
    for label, data, *_ in datasets:
        print(f"\nComputing E(k) for {label}...", flush=True)
        all_Ek[label], all_Ek_sd[label] = compute_Ek_trajectory(data)

    T  = u_gt.shape[1]
    nk = all_Ek['Ground Truth'].shape[1]
    ks = _np.arange(nk)

    # ------------------------------------------------------------------ #
    # Metric 1: Time-averaged E(k) — GT vs NSM vs UNet on one plot
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for label, _, color, ls, lw in datasets:
        Ek_mean = all_Ek[label].mean(axis=0)     # (nk,) mean over time and batch
        ax1.loglog(ks[1:], Ek_mean[1:], color=color, linestyle=ls,
                   linewidth=lw, label=label)

    ax1.set_xlabel(r'$k$',    fontsize=12)
    ax1.set_ylabel(r'$E(k)$', fontsize=12)
    ax1.set_title('Time-averaged Energy Spectrum (±1σ over time)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.2)
    fig1.tight_layout()
    out1 = OUT_PATH.replace('.png', '_time_averaged.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved {out1}", flush=True)
    plt.close(fig1)

    # ------------------------------------------------------------------ #
    # Metric 2: Final snapshot E(k) ± σ across ensemble
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for label, _, color, ls, lw in datasets:
        Ek_final = all_Ek[label][-1]             # (nk,) ensemble mean at t=T
        ax2.loglog(ks[1:], Ek_final[1:], color=color, linestyle=ls,
                   linewidth=lw, label=label)

    ax2.set_xlabel(r'$k$',    fontsize=12)
    ax2.set_ylabel(r'$E_T(k)$', fontsize=12)
    ax2.set_title(f'Final Snapshot Energy Spectrum $t=T$ (±1σ over ensemble)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', alpha=0.2)
    fig2.tight_layout()
    out2 = OUT_PATH.replace('.png', '_final_snapshot.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved {out2}", flush=True)
    plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Metric 3: RALSD(t) — spectral error over time
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ts = _np.arange(T)

    for label, _, color, ls, lw in datasets:
        if label == 'Ground Truth':
            continue
        r = ralsd(all_Ek[label], all_Ek['Ground Truth'])   # (time,)
        ax3.plot(ts, r, color=color, linestyle=ls, linewidth=lw, label=label)
    
        # mean over trajectory
        ralsd_scalar = ralsd(all_Ek[label], all_Ek['Ground Truth']).mean()
        ralsd_final  = ralsd(all_Ek[label], all_Ek['Ground Truth'])[-1]
        print(f"{label} RALSD: {ralsd_scalar:.4f}", flush=True)
        print(f"{label} RALSD at t=T: {ralsd_final:.4f}", flush=True)

    ax3.set_xlabel('Timestep $t$',  fontsize=12)
    ax3.set_ylabel('RALSD',         fontsize=12)
    ax3.set_title('Radially Averaged Log Spectral Distance over Time', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2)
    fig3.tight_layout()
    out3 = OUT_PATH.replace('.png', '_ralsd.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved {out3}", flush=True)
    plt.close(fig3)


if __name__ == '__main__':
    plot_spectra_comparison()