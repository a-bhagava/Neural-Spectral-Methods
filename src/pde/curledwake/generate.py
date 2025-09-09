from . import *
import mitwindfarm
import jax
from tqdm import tqdm

from mitwindfarm.Windfield import Uniform
from mitwindfarm.CurledWake import CurledWakeWindfield
from jax import random

def simulate_curledwake_trajectory(_u0, X=10, dx=0.1, nX=80, nu=1e-3):

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

    _v0, _w0 = velocity_field(u=_u0)

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
    return snapshots


def generate(pde: CurledWake, dx: float = 0.1, X: int = 50, Y: int = 128):

    params = pde.params.sample(random.PRNGKey(0), (128, )) # draw 128 random samples for params (u4/thrust coefficient conditions)
    solve = F.partial(simulate_curledwake_trajectory, X=pde.X, dx=dx, nX=X, nu=pde.nu)

    iterable_params = jax.vmap(lambda _u: _u.to(1, Y, Y).inv().squeeze())(params)

    _u = []
    for param in tqdm(iterable_params):
        reshape_param = param[None, ...]
        result = solve(reshape_param)
        _u.append(result)

    _u = np.stack(_u) 
    _u = np.pad(_u, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)
    np.save(f"{dir}/_u_ic.npy", params.coef)
    np.save(f"{dir}/_u_full.npy", _u)

    return _u

if __name__ == "__main__":
    generate(wake_re3)