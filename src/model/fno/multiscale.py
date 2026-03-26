from .. import *
from .._base import *
from ...basis import *
from ...basis.fourier import *
from ...basis.chebyshev import *
from ...basis.multiscalefourier import MSF
import jaxwt as jwt

# TODO: 
# double check the math on the MSF offset class vs the fourier class (e.g. make sure the frequencies are aligned correctly and that the basis functions are orthogonal)
# write an algorithm to identify the correct fourier bases to use for each wavelet scale (e.g. by looking at the frequency spectrum of the wavelet coefficients and choosing the fourier bases that correspond to those frequencies)

def zero_triplet(triplet):
    cH, cV, cD = triplet
    return (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))

def reconstruct_coeffs(coeffs, level, component, wav):
    """component = 'approx' or level number (1..level)"""
    if component == "approx":
        new_coeffs = [coeffs[0]] + [zero_triplet(t) for t in coeffs[1:]]
    else:
        new_coeffs = [np.zeros_like(coeffs[0])]
        for i in range(1, level+1):
            if i == component:
                new_coeffs.append(coeffs[i])
            else:
                new_coeffs.append(zero_triplet(coeffs[i]))
    return new_coeffs


class FNO(Spectral):

    def __repr__(self): return "NSM"

    @nn.compact
    def forward(self, ϕ: Basis) -> Basis:
        wavelet = self.cfg.get("wavelet", "haar")
        levels  = self.cfg.get("wavelet_levels", 2)

        # build one basis per level + one for approx
        # e.g. levels=2 → [T_approx, T_detail1, T_detail2]
        bases = [self.pde.basis] + [
                series(Chebyshev, MSF(offset), MSF(offset))
                for offset in self.cfg["msf_offsets"]
            ]

        # decompose input
        coeffs = jwt.wavedec2(ϕ.inv()[0, :, :, 0], wavelet, level=levels)

        # reconstruct each subband in physical space
        subbands = []
        approx_coeffs = reconstruct_coeffs(coeffs, levels, "approx", wavelet)
        subbands.append(jwt.waverec2(approx_coeffs, wavelet)[:, :, :, np.newaxis])
        for i in range(1, levels + 1):
            detail_coeffs = reconstruct_coeffs(coeffs, levels, i, wavelet)
            subbands.append(jwt.waverec2(detail_coeffs, wavelet)[:, :, :, np.newaxis])

        # project each subband onto its matched basis
        us = []
        for i, (subband, T) in enumerate(zip(subbands, bases)):
            x = np.broadcast_to(subband, ϕ.coef.shape)
            u = T.transform(x).to(*self.cfg["mode"])

            bias = T.transform(u.grid(*u.mode)).coef
            u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))

            u = u.map(nn.Dense(self.cfg["hdim"] * 4))
            u = T.transform(self.activate(u.inv()))
            u = u.map(nn.Dense(self.cfg["hdim"]))
            u = T.transform(self.activate(u.inv()))
            us.append((u, T))

        # parallel FNO layers
        for _ in range(self.cfg["depth"]):
            new_us = []
            for u, T in us:
                conv = SpectralConv(self.cfg["hdim"])(u)
                fc   = u.map(nn.Dense(self.cfg["hdim"]))
                u    = T.add(conv, fc)
                u    = T.transform(self.activate(u.inv()))
                new_us.append((u, T))
            us = new_us

        # fusion
        outs = [u.inv() for u, _ in us]
        u_cat = np.concatenate(outs, axis=-1)
        u = bases[0].transform(u_cat)

        # decoder
        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = bases[0].transform(self.activate(u.inv()))
        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = bases[0].transform(self.activate(u.inv()))
        u = u.map(nn.Dense(self.pde.odim))
        return self.pde.mollifier(ϕ, u)
