from .. import *
from ...dists import *
from ...basis import *

from .._domain import *
from .._params import *

from ...basis.fourier import *
from ...basis.chebyshev import *
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve

# TODO: double check PDE implementation 
# TODO: spectral analysis of the solution -- get a sense of which fourier modes / chebyshev modes to actually try here
# TODO: run hyperparameter sweep for preliminary experiments; get insights and finetune on these results
# TODO: performance analysis across different thrust coefficients / u4 values

def gaussian_kernel(size_y: int, size_z: int, sigma_y: float, sigma_z: float):
    """Create a separable 2D Gaussian kernel (JAX)."""
    y = jnp.arange(-size_y // 2 + 1., size_y // 2 + 1.)
    z = jnp.arange(-size_z // 2 + 1., size_z // 2 + 1.)
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    kernel = jnp.exp(-(yy**2 / (2*sigma_y**2) + zz**2 / (2*sigma_z**2)))
    return kernel / jnp.sum(kernel)


def gaussian_blur(image, sigma_y, sigma_z, size=21):
    # fixed kernel size (Python int, hashable)
    y = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    z = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    yy, zz = jnp.meshgrid(y, z, indexing="ij")
    
    kernel = jnp.exp(-(yy**2 / (2*sigma_y**2) + zz**2 / (2*sigma_z**2)))
    kernel = kernel / jnp.sum(kernel)
    
    return fftconvolve(image, kernel, mode="same")


class WakeInitial(Uniform):

    grid = Fourier[2].grid(128, 128)

    def __str__(self): return f"u4={round(self.rotor_u4, 3)}"

    def __init__(self, grid_shape=(128, 128),
                 turbine_y: float = 0.0, turbine_z: float = 0.0,
                 rotor_diameter: float = 1.0, u4_min: float = 0.25, 
                 u4_max: float = 0.75, REWS: float = 1.0, 
                 smooth_fact: float = 0.1): 
        
        super().__init__(u4_min, u4_max)       
        self.grid_shape = grid_shape
        self.turbine_y = turbine_y
        self.turbine_z = turbine_z
        self.rotor_diameter = rotor_diameter
        self.rotor_u4 = jnp.mean(jnp.array([u4_max, u4_min]))
        self.rotor_REWS = REWS
        self.smooth_fact = smooth_fact
        
        # Create coordinate grids
        self.y_coords = jnp.linspace(-2, 2, grid_shape[0])
        self.z_coords = jnp.linspace(-2, 2, grid_shape[1])

    def wake_deficit_ic(self, y, z, yt, zt, smooth_fact, ay, az):
        """ Create turbine wake stencil represented by a gaussian smoothed indicator function """
        # define azimuthal radius if not provided
        az = ay if az is None else az
        # create meshgrid for y and z
        yG, zG = jnp.meshgrid(y, z, indexing="ij")
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # calculate normalized distance from center
        dist = jnp.sqrt(((yG - yt) / ay) ** 2 + ((zG - zt) / az) ** 2)

        # create smooth mask using tanh transition
        mask = 0.5 * (1 - jnp.tanh((dist - 1) / (1e-8 + smooth_fact * 0.1)))

        # apply Gaussian smoothing
        sigma_y = smooth_fact / dy  # grid units in y-direction
        sigma_z = smooth_fact / dz  # grid units in z-direction
        result = gaussian_blur(mask, sigma_y, sigma_z)
        return result

    def sample(self, prng, shape=()):
        # base field = all ones
        # base_field = np.ones(shape + self.grid_shape)  # (…, Ny, Nz)

        # turbine wake deficit (same for all samples)
        r4 = self.rotor_diameter / 2
        wake_deficit = self.wake_deficit_ic(
            self.y_coords, self.z_coords,
            self.turbine_y, self.turbine_z,
            smooth_fact=self.smooth_fact,
            ay=r4, az=r4
        )  # (Ny, Nz)

        # sample u4 values
        u4_sampled = super().sample(prng, shape)
        delta_u = u4_sampled - self.rotor_REWS
        delta_u = delta_u[(...,) + (None,) * len(self.grid_shape)]

        wake_field = delta_u * wake_deficit 
        ic_scalar = wake_field # + base_field  

        # expand to 2 components
        ic = ic_scalar[..., np.newaxis, :, :]
        ic = np.broadcast_to(ic, shape + (2, *self.grid_shape))
        return ic
    

def velocity_field(u=None,
                   D=1,
                   N_vortex=128,
                   sigma_vortex=0.2,
                   turbine_y=0.0, 
                   turbine_z=0.0,):

    if len(u.shape) == 2: 
        ydim, zdim = u.shape
    elif len(u.shape) == 3:
        _, ydim, zdim = u.shape

    y_coords = np.linspace(-2, 2, ydim)
    z_coords = np.linspace(-2, 2, zdim)
    dz = z_coords[1] - z_coords[0]

    # clip edges to prevent singularities
    r_i = np.linspace(-(D - dz) / 2, (D - dz) / 2, N_vortex)

    # calculate total circulation from the yaw 
    Gamma_0 = 0.5 * D * 1 * 2 * np.sin(30 * np.pi / 180)
    Gamma_i = (
            Gamma_0 * 4 * r_i / (N_vortex * D**2 * np.sqrt(1 - (2 * r_i / D) ** 2))
        )
    sigma = sigma_vortex * D

    # main summation, which is 3D (y, z, i)
    yG, zG = np.meshgrid(y_coords, z_coords, indexing="ij")
    yG = yG[..., None]
    zG = zG[..., None]
    rsq = (yG - turbine_y) ** 2 + (zG - turbine_z - r_i[None, None, :]) ** 2  # 3D grid variable
    rsq = np.clip(rsq, 1e-8, None)  # avoid singularities
    exponent = 1 - np.exp(-rsq / sigma**2)
    summation = exponent / (2 * np.pi * rsq) * Gamma_i[None, None, :]

    # sum all vortices along last dim
    v_slice = np.sum(summation * (zG - turbine_z - r_i[None, None, :]), axis=-1)
    w_slice = np.sum(summation * -(yG - turbine_y), axis=-1)

    # tile along the x direction 
    v = np.broadcast_to(v_slice, (u.shape[0], *v_slice.shape))
    w = np.broadcast_to(w_slice, (u.shape[0], *w_slice.shape))    
    return v, w


class CurledWake(PDE):
    def __str__(self):
        return f"CurledWake: X={self.X}, ν={self.nu}"
    
    def __init__(self, ic: WakeInitial, X: float, nu: float): 

        self.odim = 1
        self.ic = ic

        self.X = X
        self.nu = nu
        self.fn = None  # no forcing term for now

        self.domain = Rect(3)
        self.basis = series(Chebyshev, Fourier, Fourier)
        self.params = Interpolate(ic, self.basis)

        from ..mollifier import initial_condition
        self.mollifier = initial_condition

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        with jax.default_device(jax.devices("cpu")[0]):

            _u = np.load(f"{dir}/_u_ic.npy")
            _u_full = np.load(f"{dir}/_u_full.npy")
            
        return jax.vmap(self.basis)(_u), _u_full.shape[1:-1], _u_full

    def equation(self, x: X, _u0: X, _u: X): 
        _u, _u1, _u2 = utils.fdm(_u, n=2)
        _ux = _u1[..., 0, 0]
        _uy = _u1[..., 0, 1]
        _uz = _u1[..., 0, 2]
        _Δu = Δ(_u2[..., 0, 1:, 1:])

        v, w = velocity_field(u=_u.squeeze(-1))

        # base flow (U=1, V, W=0)
        U = 1.0  

        # advection terms
        adv = (U + _u.squeeze(-1)) * (_ux) + (v * _uy + w * _uz)

        # forcing term
        if self.fn is None: f = np.zeros_like(adv)
        else: f = self.fn(*_u[0].squeeze(-1).shape)

        return adv + self.nu * _Δu + f

    def boundary(self, _u: List[Tuple[X]])-> List[X]: 
        _, (_ut, _ub), (_ul, _ur) = _u
        return [_ut - _ub, _ul - _ur]

    def spectral(self, _u0: Basis, _u: Basis) -> Basis:
        _u1 = _u.grad()
        _u2 = _u1.grad()

        _ux = self.basis(_u1.coef[..., 0, 0])
        _uy = self.basis(_u1.coef[..., 0, 1])
        _uz = self.basis(_u1.coef[..., 0, 2])
        _Δu = self.basis(Δ(_u2.coef[..., 0, 1:, 1:]))

        v, w = velocity_field(u=_u.inv().squeeze(-1))

        # base flow (U=1, V=W=0)
        U = 1.0  

        # advection term: (U+u) ∂u/∂x + v ∂u/∂y + w ∂u/∂z
        scaled_udef = self.basis.mul(self.basis.transform(U + _u.inv().squeeze(-1)), _ux)
        adv = self.basis.add(scaled_udef, self.basis.transform(v * _uy.inv() + w * _uz.inv()))

        # forcing term
        if self.fn is None: f = self.basis(np.zeros_like(adv.coef))
        else: f = self.basis.transform(np.broadcast_to(self.fn(*_u.mode[1:]), _u.mode))

        # final PDE: advection + ν Δu + f
        return self.basis.add(adv, self.basis(self.nu * _Δu.coef), f)

# ------------------------------ INITIAL CONDITION ----------------------------- #

ic = WakeInitial()

# ------------------------------- CURLED WAKE ------------------------------ #

wake_re3 = CurledWake(ic, X=10, nu=1e-3)