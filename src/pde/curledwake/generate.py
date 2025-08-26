from . import *

def simulate(): 
    pass 


def solution():
    pass


def generate(pde: CurledWake, dt: float = 1e-3, T: int = 64, X: int = 256):

    params = pde.params.sample(random.PRNGKey(0), (128, )) # 128 random samples for params, uniformly sampled u4/thrust coefficient conditions 
    solve = F.partial(solution, T=pde.T, nu=pde.nu, force=pde.fn, dt=dt, nt=T) # instantiate the solver 

    _u = jax.vmap(solve)(jax.vmap(lambda _u: _u.to(1, X, X).inv().squeeze())(params))
    _u = np.pad(_u, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)

    np.save(f"{dir}/_u_ic.{pde.ic}.npy", params.coef)
    np.save(f"{dir}/_u_full.{pde}.npy", _u)

    return _u

if __name__ == "__main__":
    pass