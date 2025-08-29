from . import *
import mitwindfarm
import jax
from tqdm import tqdm

from mitwindfarm.Windfield import Uniform
from mitwindfarm.CurledWake import CurledWakeWindfield
from jax import random

# things to check 
    # do the inputs u, v, w make sense -- should velocity_field be called before or after the subtraction of 1 from u0?
    # do i need to add 1 back to the output udef to get the actual velocity field u? -- then should i clip the baseflow at 1? 
    # correct the self.basis.mul in the curledwake pde class
    # correct the naming convention in the ic class / pde class so the file saving is cleaner


def simulate_curledwake(_u0, X=10, nu=1e-3, dx=0.1,):

    curledwake_solver_args = dict(
            dx=dx,
            dy=4 / 127,
            dz=4 / 127,
            integrator="rk4",  # see mitwindfarm.utils.integrate
            k_model="const",  # alternatives: "const", "2021"
            k_kwargs = dict(nu_T=nu),
            ybuff=2, 
            zbuff=2,
            N_vortex=128,
            sigma_vortex=0.2,
            auto_expand=False,
            verbose=False,
        )

    _udef = _u0 - 1
    _v0, _w0 = velocity_field(u=_udef)

    base_windfield = Uniform() 
    curled_wake_windfield = CurledWakeWindfield(base_windfield, **curledwake_solver_args)
    curled_wake_windfield.check_grid_init(x=0, y=0, z=0)

    # stamp the initial condition on the windfield
    curled_wake_windfield.du = _udef
    curled_wake_windfield.dv = _v0
    curled_wake_windfield.dw = _w0

    # march to the desired spatial location
    curled_wake_windfield.march_to(x=X, y=0, z=0)

    return curled_wake_windfield.du + 1  # add back the base windfield


def generate(pde: CurledWake, dx: float = 0.1, X: int = 10, Y: int = 128):

    params = pde.params.sample(random.PRNGKey(0), (128, )) # draw 128 random samples for params (u4/thrust coefficient conditions)
    solve = F.partial(simulate_curledwake, X=pde.X, nu=pde.nu, dx=dx)

    iterable_params = jax.vmap(lambda _u: _u.to(1, Y, Y).inv().squeeze())(params)

    _u = []
    for param in tqdm(iterable_params):
        reshape_param = param[None, ...]
        result = solve(reshape_param)
        _u.append(result)

    _u = np.stack(_u) 
    _u = np.pad(_u, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)
    np.save(f"{dir}/_u_ic.{pde.ic}.npy", params.coef)
    np.save(f"{dir}/_u_full.{pde}.npy", _u)

    return _u

if __name__ == "__main__":
    generate(wake_re3)