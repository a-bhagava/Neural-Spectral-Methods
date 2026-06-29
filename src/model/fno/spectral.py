from .. import *
from .._base import *
from ...basis.fourier import *

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class FNO(Spectral):

    def __repr__(self): return "NSM"

    @nn.compact
    def forward(self, ϕ: Basis, apply_noise: bool = False) -> Basis:

        if not self.cfg["fourier"]: T = self.pde.basis
        else: T = Fourier[self.pde.domain.ndim]

        u = ϕ.to(*self.cfg["mode"])

        if apply_noise:
            # apply data augmentation if specified 
            uniform_gaussian_noise = self.cfg.get("uniform_gaussian_noise")
            energy_scaled_gaussian_noise = self.cfg.get("energy_scaled_gaussian_noise")
            complementary_spectral_dropout = self.cfg.get("spectral_dropout")

            # for dropout, apply bernouli mask - stochastic spectral mask with prob b
            def apply_dropout(c):
                dropout_rate = self.cfg.get("spectral_dropout_rate", 0.1)
                rng = self.make_rng("noise")
                mask = jax.random.bernoulli(rng, p=1 - dropout_rate, shape=c.shape[:-1])
                mask = mask[..., None]
                return c * mask

            if uniform_gaussian_noise:
                sigma = self.cfg["spectral_noise_std"]
                def add_uniform_noise(c):
                    rng = self.make_rng("noise")
                    noise = sigma * jax.random.normal(rng, shape=c.shape, dtype=c.dtype)
                    out = c + noise
                    return out
                u = u.map(add_uniform_noise)
                if complementary_spectral_dropout:
                    u = u.map(apply_dropout)

            if energy_scaled_gaussian_noise:
                sigma = self.cfg["spectral_noise_std"]
                def add_energy_noise(c):
                    rng = self.make_rng("noise")
                    noise = sigma * jax.random.normal(rng, shape=c.shape, dtype=c.dtype)
                    out = c * (1 + noise)
                    return out
                u = u.map(add_energy_noise)

                if complementary_spectral_dropout:
                    u = u.map(apply_dropout)

        bias = T.transform(u.grid(*u.mode)).coef
        u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = T.transform(self.activate(u.inv()))

        for _ in range(self.cfg["depth"]):

            conv = SpectralConv(self.cfg["hdim"])(u)
            fc = u.map(nn.Dense(self.cfg["hdim"]))

            u = T.add(conv, fc)
            u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.pde.odim))
        return self.pde.mollifier(ϕ, u)
