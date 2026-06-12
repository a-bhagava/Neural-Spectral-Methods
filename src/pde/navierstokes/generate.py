# Modified from neuraloperator. Commit ef3de3bb1140175c69a9fe3a8b45afd1335077d9
# https://github.com/neuraloperator/neuraloperator/blob/master/data_generation/navier_stokes/ns_2d.py

from . import *

def simulate(w0: X, nu: float, f: X) -> Fx:

    """
    Returns:
        u: callable what -> what' for next step
           advance vorticity in spectral domain
    """
    s, s = w0.shape
    kx, ky = k(s, s)

    diffuse = (Δ := kx ** 2 + ky ** 2) * nu

    dealias = (Δ < (2/3 * π * s) ** 2).astype(float)
    dealias = dealias.at[0, 0].set(0)   # zero-mean

    def Δhat(what: X) -> X: # defines the operator Δ in spectral domain

        vx, vy = velocity(what)

        wx = np.fft.irfft2(what * 1j * kx, (s, s))
        wy = np.fft.irfft2(what * 1j * ky, (s, s))

        vxwx = np.fft.fft2(vx * wx, (s, s))
        vywy = np.fft.fft2(vy * wy, (s, s))

        return np.fft.fft2(f) - vxwx - vywy

    def call(what: X, dt: float) -> X:
        
        Δhat1 = Δhat(what)  # Heun's method -- predictor 

        what_tilde = what + dt * (Δhat1 - diffuse * what / 2)
        what_tilde/= 1 + dt * diffuse / 2

        Δhat2 = Δhat(what_tilde)  # Cranck-Nicholson + Heun -- corrector

        what = what + dt * ((Δhat1 + Δhat2) - diffuse * what) / 2
        what/= 1 + dt * diffuse / 2

        return what * dealias

    return call

def solution(w0: X, T: float, nu: float, force: Fx,
             dt: float, nt: int) -> X:

    """
    Args:
        w0: initial condition

        T: total time
        nu: viscosity
        force: -ing term

        dt: advance step
        nt: record step

    Returns:
        u: solution recorded at each timestep
           inclusive of the end time
           shape = (nt, *w0.shape)
    """

    if not force: f = np.zeros_like(w0)
    else: f = force(*w0.shape)

    step = simulate(w0, nu, f) # returns callable to advance vorticity in spectral domain
    Δt = T / (N := nt - 1)

    def record(what: X, _) -> Tuple[X, X]:
        call = lambda _, what: step(what, dt)

        what = step(jax.lax.fori_loop(0., Δt // dt, call, what), Δt % dt)
        return what, np.fft.irfft2(what, s=w0.shape)

    _, w = jax.lax.scan(record, np.fft.fft2(w0), None, N) # jax primitive for loops 
    return np.concatenate([w0[np.newaxis, :], w], axis=0) # stack initial condition with the other solutions

# ---------------------------------------------------------------------------- #
#                                   GENERATE                                   #
# ---------------------------------------------------------------------------- #

def generate(pde: NavierStokes, dt: float = 1e-3, T: int = 64, X: int = 256):

    params = pde.params.sample(random.PRNGKey(0), (128, )) # 128 random samples for params, gaussian random fields
    solve = F.partial(solution, T=pde.T, nu=pde.nu, force=pde.fn, dt=dt, nt=T) # instantiate the solver

    w = jax.vmap(solve)(jax.vmap(lambda w: w.to(1, X, X).inv().squeeze())(params)) # solve the PDE for each initial condition 
    w = np.pad(w, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)

    np.save(f"{dir}/w.{pde.ic}.npy", params.coef)
    np.save(f"{dir}/u.{pde}.npy", w)

    return w


def generate_mixed(pdes: list, n_per: int = None, seed: int = 0):
    """
    Combines pre-generated data from multiple PDEs into a single mixed dataset.
    Subsamples equally across scales. Saves in the same format as generate().

    Args:
        pdes:    list of NavierStokes instances (e.g. [re4_2, re4_4, re4_8])
        n_per:   samples per scale (default: min available across all scales)
        seed:    random seed for subsampling
    """

    dir = os.path.dirname(__file__)

    all_w, all_u = [], []
    for pde in pdes:
        w_path = f"{dir}/w.{pde.ic}.npy"
        u_path = f"{dir}/u.{pde}.npy"
        all_w.append(np.load(w_path))
        all_u.append(np.load(u_path))
        print(f"  Loaded {pde.ic}: w={all_w[-1].shape}  u={all_u[-1].shape}")

    # default to balanced split at the smallest scale
    if n_per is None:
        n_per = min(w.shape[0] for w in all_w) // len(pdes)
    print(f"  Subsampling {n_per} per scale ({n_per * len(pdes)} total)")

    rng = key = jax.random.PRNGKey(seed)
    sub_w, sub_u = [], []

    for w, u in zip(all_w, all_u):
        idx = jax.random.permutation(rng, w.shape[0])[:n_per]
        sub_w.append(w[idx])
        sub_u.append(u[idx])

    w_cat = np.concatenate(sub_w, axis=0)
    u_cat = np.concatenate(sub_u, axis=0)

    # name reflects the mix, e.g. "0.2x1.0+0.4x1.0+0.8x1.0" and "Re=800:T=3:None+..."
    ic_name  = "+".join(str(pde.ic)  for pde in pdes)
    pde_name = "+".join(str(pde)     for pde in pdes)

    w_out = f"{dir}/w.{ic_name}.npy"
    u_out = f"{dir}/u.{pde_name}.npy"

    np.save(w_out, w_cat)
    np.save(u_out, u_cat)
    print(f"  Saved w → {w_out}  shape={w_cat.shape}")
    print(f"  Saved u → {u_out}  shape={u_cat.shape}")

    return w_cat, u_cat


if __name__ == "__main__":

    from sys import argv
    
    if argv[1] == "ns":
        generate(re2)
        generate(re3)
        generate(re4)

    if argv[1] == "tf":
        generate(tf, dt=5e-3)

    if argv[1] == "mixed":
        generate_mixed([re4_2, re4_4, re4_8], n_per=43)  # 43 per scale → 129 total (fits in memory/approx equal to 128)
        generate_mixed([re4_2, re4_8], n_per=64)  # 64 per scale → 128 total (fits in memory/approx equal to 128)
        generate_mixed([re4_4, re4_8], n_per=64)  # 64 per scale → 128 total (fits in memory/approx equal to 128)