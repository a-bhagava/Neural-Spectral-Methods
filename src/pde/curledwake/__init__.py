from .. import *
from ...dists import *
from ...basis import *

from .._domain import *
from .._params import *

from ...basis.fourier import *
from ...basis.chebyshev import *
from scipy.ndimage import gaussian_filter


class WakeInitial(Uniform):

    grid = Fourier[2].grid(128, 128)

    def __str__(self): return f"{self.grid_shape}x{self.rotor_u4}"

    def __init__(self, grid_shape=(128, 128),
                 turbine_y: float = 0.0, turbine_z: float = 0.0,
                 rotor_diameter: float = 1.0, u4_min: float = 0.15, 
                 u4_max: float = 0.95, REWS: float = 1.0, 
                 smooth_fact: float = 0.1): 
        
        super().__init__(u4_min, u4_max)       
        self.grid_shape = grid_shape
        self.turbine_y = turbine_y
        self.turbine_z = turbine_z
        self.rotor_diameter = rotor_diameter
        self.rotor_u4 = np.mean(np.array([u4_max, u4_min]))
        self.rotor_REWS = REWS
        self.smooth_fact = smooth_fact
        
        # Create coordinate grids
        self.y_coords = np.linspace(-2, 2, grid_shape[0])
        self.z_coords = np.linspace(-2, 2, grid_shape[1])

    def wake_deficit_ic(self, y, z, yt, zt, smooth_fact, ay, az):
        """ Create turbine wake stencil represented by a gaussian smoothed indicator function """
        # define azimuthal radius if not provided
        az = ay if az is None else az
        # create meshgrid for y and z
        yG, zG = np.meshgrid(y, z, indexing="ij")
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # calculate normalized distance from center
        dist = np.sqrt(((yG - yt) / ay) ** 2 + ((zG - zt) / az) ** 2)

        # create smooth mask using tanh transition
        mask = 0.5 * (1 - np.tanh((dist - 1) / (1e-8 + smooth_fact * 0.1)))

        # apply Gaussian smoothing
        sigma_y = smooth_fact / dy  # grid units in y-direction
        sigma_z = smooth_fact / dz  # grid units in z-direction
        result = gaussian_filter(mask, sigma=[sigma_y, sigma_z])
        return result

    def sample(self, prng, shape=()):
        # base field = all ones
        base_field = np.ones(shape + self.grid_shape)  # (…, Ny, Nz)

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
        ic_scalar = base_field + wake_field

        # expand to 2 components
        ic = ic_scalar[..., np.newaxis, :, :]
        ic = np.broadcast_to(ic, shape + (2, *self.grid_shape))
        return ic
    

def velocity_field(u=None,
                   D=1,
                   N_vortex=32,
                   sigma_vortex=0.1,
                   turbine_y=0.0, 
                   turbine_z=0.0,):
    
    y_coords = np.linspace(-2, 2, u.shape[1])
    z_coords = np.linspace(-2, 2, u.shape[2])
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

        self.l = (l:=ic.length)
        self.Re = l / nu * ic.scaling

        self.domain = Rect(3)
        self.basis = series(Chebyshev, Fourier, Fourier)
        self.params = Interpolate(ic, self.basis)

        from ..mollifier import initial_condition
        self.mollifier = initial_condition

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        with jax.default_device(jax.devices("cpu")[0]):

            _u = np.load(f"{dir}/_u.{self.ic}.npy")
            u_full = np.load(f"{dir}/u.{self}.npy") # TODO: check this makes sense after data generation scripts are written
            
        return jax.vmap(self.basis)(_u), u_full.shape[1:-1], u_full

    def equation(self, x: X, _u0: X, _u: X): 
        _u, _u1, _u2 = utils.fdm(_u, n=2)
        _ux = _u1[..., 0, 0]
        _uy = _u1[..., 0, 1]
        _uz = _u1[..., 0, 2]
        _Δu = Δ(_u2[..., 0, 1:, 1:])

        v, w = velocity_field(u=_u.squeeze(-1))
        _Dudx = _ux / self.X + (v * _uy + w * _uz) # self.X is the scaling factor TODO: verify with kirby, division by U

        if self.fn is None: f = np.zeros_like(_Dudx)
        else: f = self.fn(*_u[0].squeeze(-1).shape)

        return (_Dudx + self.nu * _Δu + f) # TODO: check with kirby about the -1/U division and the addition/subtraction signs here

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
        _Dudx = self.basis.add(
            _ux.map(lambda coef: coef / self.X),
            self.basis.transform(v * _uy.inv() + w * _uz.inv())
        )
        
        # forcing term
        if self.fn is None: f = self.basis(np.zeros_like(_Dudx.coef))
        else: f = self.basis.transform(np.broadcast_to(self.fn(*_u.mode[1:]), _u.mode))

        return self.basis.add(_Dudx, self.basis(self.nu * _Δu.coef), f.map(np.negative)) # TODO: check with kirby about the -1/U division and the addition/subtraction signs here