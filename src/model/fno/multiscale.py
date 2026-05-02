from .. import *
from .._base import *
from ...basis import *
from ...basis.fourier import *
from ...basis.chebyshev import *
from ...basis.multiscalefourier import MSF
import jaxwt as jwt

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


# class FNO(Spectral):

#     def __repr__(self): return "NSM"

#     @nn.compact
#     def forward(self, ϕ: Basis) -> Basis:
#         wavelet = self.cfg.get("wavelet", "haar")
#         levels  = self.cfg.get("wavelet_levels", 2)

#         # build one basis per level + one for approx
#         # e.g. levels=2 → [T_approx, T_detail1, T_detail2]
#         bases = [self.pde.basis] + [
#                 series(Chebyshev, MSF(offset), MSF(offset))
#                 for offset in self.cfg["msf_offsets"]
#             ]

#         # decompose input
#         coeffs = jwt.wavedec2(ϕ.inv()[0, :, :, 0], wavelet, level=levels)

#         # reconstruct each subband in physical space
#         subbands = []
#         approx_coeffs = reconstruct_coeffs(coeffs, levels, "approx", wavelet)
#         subbands.append(jwt.waverec2(approx_coeffs, wavelet)[:, :, :, np.newaxis])
#         for i in range(1, levels + 1):
#             detail_coeffs = reconstruct_coeffs(coeffs, levels, i, wavelet)
#             subbands.append(jwt.waverec2(detail_coeffs, wavelet)[:, :, :, np.newaxis])

#         # project each subband onto its matched basis
#         us = []
#         for i, (subband, T) in enumerate(zip(subbands, bases)):
#             x = np.broadcast_to(subband, ϕ.coef.shape)
#             u = T.transform(x).to(*self.cfg["mode"])

#             bias = T.transform(u.grid(*u.mode)).coef
#             u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))

#             u = u.map(nn.Dense(self.cfg["hdim"] * 4))
#             u = T.transform(self.activate(u.inv()))
#             u = u.map(nn.Dense(self.cfg["hdim"]))
#             u = T.transform(self.activate(u.inv()))
#             us.append((u, T))

#         # parallel FNO layers
#         for _ in range(self.cfg["depth"]):
#             new_us = []
#             for u, T in us:
#                 conv = SpectralConv(self.cfg["hdim"])(u)
#                 fc   = u.map(nn.Dense(self.cfg["hdim"]))
#                 u    = T.add(conv, fc)
#                 u    = T.transform(self.activate(u.inv()))
#                 new_us.append((u, T))
#             us = new_us

#         # fusion
#         outs = [u.inv() for u, _ in us]
#         u_cat = np.concatenate(outs, axis=-1)
#         u = bases[0].transform(u_cat)

#         # decoder
#         u = u.map(nn.Dense(self.cfg["hdim"]))
#         u = bases[0].transform(self.activate(u.inv()))
#         u = u.map(nn.Dense(self.cfg["hdim"] * 4))
#         u = bases[0].transform(self.activate(u.inv()))
#         u = u.map(nn.Dense(self.pde.odim))
#         return self.pde.mollifier(ϕ, u)


class FNO(Spectral):

    def __repr__(self): return "NSM"

    @nn.compact
    def forward(self, ϕ: Basis) -> Basis:
        wavelet = self.cfg.get("wavelet", "haar")
        levels  = self.cfg.get("wavelet_levels", 2)

        # Parse MSF configuration: "offsets|modes"
        # Example: "0 0 21 21 43 43|21 21 22 22 21 21"
        msf_config = self.cfg.get("msf_config", None)
        
        if msf_config is not None:
            # New approach: parse msf_config string
            offsets_str, modes_str = msf_config.split('|')
            offsets_flat = list(map(int, offsets_str.split()))
            modes_flat = list(map(int, modes_str.split()))
            
            # Parse into (y, z) pairs
            offsets = [(offsets_flat[i], offsets_flat[i+1]) for i in range(0, len(offsets_flat), 2)]
            modes = [(modes_flat[i], modes_flat[i+1]) for i in range(0, len(modes_flat), 2)]
            
            # Build bases per level
            bases = []
            for i, (off_y, off_z) in enumerate(offsets):
                if i == 0:
                    # Level 0 (approx): use standard Fourier (offset always 0)
                    bases.append(self.pde.basis)
                else:
                    # Detail levels: use MSF with specified offset
                    bases.append(series(Chebyshev, MSF(off_y), MSF(off_z)))
        else:
            # Legacy approach: use msf_offsets if provided
            msf_offsets = self.cfg.get("msf_offsets", [])
            if len(msf_offsets) == 0:
                # No offsets specified, use 0 for all detail levels
                msf_offsets = [0] * levels
            
            bases = [self.pde.basis] + [
                series(Chebyshev, MSF(offset), MSF(offset))
                for offset in msf_offsets
            ]
            # Use mode from cfg for all levels
            modes = [tuple(self.cfg["mode"][1:])] * (levels + 1)

        # Decompose input with wavelet transform
        coeffs = jwt.wavedec2(ϕ.inv()[0, :, :, 0], wavelet, level=levels)

        # Reconstruct each subband in physical space
        subbands = []
        
        # Level 0: approximation coefficients
        approx_coeffs = reconstruct_coeffs(coeffs, levels, "approx", wavelet)
        subbands.append(jwt.waverec2(approx_coeffs, wavelet)[:, :, :, np.newaxis])
        
        # Levels 1..L: detail coefficients
        for i in range(1, levels + 1):
            detail_coeffs = reconstruct_coeffs(coeffs, levels, i, wavelet)
            subbands.append(jwt.waverec2(detail_coeffs, wavelet)[:, :, :, np.newaxis])

        # Project each subband onto its matched basis
        us = []
        T_mode = self.cfg["mode"][0]  # Temporal mode (constant across levels)
        
        for i, (subband, T) in enumerate(zip(subbands, bases)):
            x = np.broadcast_to(subband, ϕ.coef.shape)
            
            # Transform and truncate to per-level mode count
            if msf_config is not None:
                # Use modes from msf_config
                mode_y, mode_z = modes[i]
                u = T.transform(x).to(T_mode, mode_y, mode_z)
            else:
                # Legacy: use mode from cfg
                u = T.transform(x).to(*self.cfg["mode"])

            # Encoder: lift to hidden dimension
            bias = T.transform(u.grid(*u.mode)).coef
            u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))
            u = u.map(nn.Dense(self.cfg["hdim"] * 4))
            u = T.transform(self.activate(u.inv()))
            u = u.map(nn.Dense(self.cfg["hdim"]))
            u = T.transform(self.activate(u.inv()))
            
            us.append((u, T))

        # Parallel FNO layers (per-branch spectral convolutions)
        for _ in range(self.cfg["depth"]):
            new_us = []
            for u, T in us:
                conv = SpectralConv(self.cfg["hdim"])(u)
                fc   = u.map(nn.Dense(self.cfg["hdim"]))
                u    = T.add(conv, fc)
                u    = T.transform(self.activate(u.inv()))
                new_us.append((u, T))
            us = new_us

        # Fusion: concatenate all branches in physical space
        outs = [u.inv() for u, _ in us]
        u_cat = np.concatenate(outs, axis=-1)
        u = bases[0].transform(u_cat)

        # Decoder: map to output dimension
        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = bases[0].transform(self.activate(u.inv()))
        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = bases[0].transform(self.activate(u.inv()))
        u = u.map(nn.Dense(self.pde.odim))
        
        return self.pde.mollifier(ϕ, u)