from . import *
import mitwindfarm
import jax
from tqdm import tqdm

from mitwindfarm.Windfield import Uniform
from mitwindfarm.CurledWake import CurledWakeWindfield
from jax import random

def simulate_curledwake_trajectory(_u0, X=10, dx=0.1, nX=80, nu=1e-3, data_mode="easy", prng=None):

    curledwake_solver_args = dict(
        dx=dx,
        dy=4 / 127,
        dz=4 / 127,
        integrator="rk4",
        k_model="const",
        k_kwargs=dict(nu_T=nu),
        ybuff=2,
        zbuff=2,
        N_vortex=128,
        sigma_vortex=0.2,
        auto_expand=False,
        verbose=False,
    )

    if data_mode == "easy": 
        sampled_yaw = 0
        _v0, _w0 = velocity_field(u=_u0, yaw=sampled_yaw)
    elif data_mode == "hard":
        max_yaw, min_yaw = 30, 1
        scale = max_yaw - min_yaw
        sampled_yaw = int(random.uniform(prng) * scale + min_yaw)
        _v0, _w0 = velocity_field(u=_u0, yaw=sampled_yaw)

    base_windfield = Uniform()
    curled_wake_windfield = CurledWakeWindfield(base_windfield, **curledwake_solver_args)
    curled_wake_windfield.check_grid_init(x=0, y=0, z=0)

    # stamp the initial condition
    curled_wake_windfield.du = _u0
    curled_wake_windfield.dv = _v0
    curled_wake_windfield.dw = _w0

    # --- recording logic ---
    Δx = X / (nX - 1)       # recording interval
    snapshots = []
        
    for i in range(nX):
        x_target = i * Δx
        curled_wake_windfield.march_to(x=x_target, y=0, z=0)
        snapshots.append(curled_wake_windfield.du[-1].copy())

    # convert to array
    snapshots = np.stack(snapshots)  # shape = (nX, Ny, Nz)
    dv0 = curled_wake_windfield.dv[-1].copy()
    dw0 = curled_wake_windfield.dw[-1].copy()
    
    return snapshots, sampled_yaw, dv0, dw0


def generate(pde: CurledWake, dx: float = 0.1, X: int = 51, Y: int = 128, data_mode="easy"):
    params = pde.params.sample(random.PRNGKey(0), (128, )) # draw 128 random samples for params (u4/thrust coefficient conditions)
    solve = F.partial(simulate_curledwake_trajectory, X=pde.X, dx=dx, nX=X, nu=pde.nu, data_mode=data_mode)

    iterable_params = jax.vmap(lambda _u: _u.to(1, Y, Y).inv().squeeze())(params)

    _u = []
    yaws = []
    v_fields = []
    w_fields = []
    
    prng = random.PRNGKey(0)
    for param in tqdm(iterable_params):
        prng, subkey = random.split(prng)
        reshape_param = param[None, ...]
        # result, yaw = solve(reshape_param, prng=subkey)
        result, yaw, dv0, dw0 = solve(reshape_param, prng=subkey)
        _u.append(result)
        yaws.append(yaw)
        v_fields.append(dv0)
        w_fields.append(dw0)

    _u = np.stack(_u) 
    v_fields = np.stack(v_fields)
    w_fields = np.stack(w_fields)
    yaws = np.stack(yaws)
    u4s = pde.ic.u4_sampled

    _u = np.pad(_u, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]
    v_fields = np.pad(v_fields, [(0, 0), (0, 1), (0, 1)], mode="wrap")
    w_fields = np.pad(w_fields, [(0, 0), (0, 1), (0, 1)], mode="wrap")

    dir = os.path.dirname(__file__)

    if data_mode in ["easy", "hard"]:
        np.save(f"{dir}/_u_ic_spectral_{data_mode}.npy", params.coef)
        np.save(f"{dir}/_u_ic_raw_{data_mode}.npy", _u[:, 0, :, :, 0])
        np.save(f"{dir}/_u_full_{data_mode}.npy", _u)
        np.save(f"{dir}/_u_ic_{data_mode}.npy", params.coef)
        np.save(f"{dir}/_yaws_{data_mode}.npy", yaws)
        np.save(f"{dir}/_u4s_{data_mode}.npy", u4s)
        np.save(f"{dir}/_v_fields_{data_mode}.npy", v_fields)
        np.save(f"{dir}/_w_fields_{data_mode}.npy", w_fields)
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    return _u

if __name__ == "__main__":
    generate(wake_re3, data_mode="hard")