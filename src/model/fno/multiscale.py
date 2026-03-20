from .. import *
from .._base import *
from ...basis import *
from ...basis.fourier import *
from ...basis.chebyshev import *
from ...basis.multiscalefourier import MSF, MSF_Mid, MSF_High
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

class FNO(Spectral):

    def __repr__(self): return "NSM"

    @nn.compact
    def forward(self, ϕ: Basis) -> Basis:
        T_approx = self.pde.basis
        T_detail1 = series(Chebyshev, MSF_Mid, MSF_Mid)
        T_detail2 = series(Chebyshev, MSF_High, MSF_High)
        
        coeffs = jwt.wavedec2(ϕ.inv()[0, :, :, 0], "haar", level=2)
        approx_coeffs = reconstruct_coeffs(coeffs, 2, "approx", "haar")
        detail1_coeffs = reconstruct_coeffs(coeffs, 2, 1, "haar")
        detail2_coeffs = reconstruct_coeffs(coeffs, 2, 2, "haar")

        approx = jwt.waverec2(approx_coeffs, "haar")[:, :, :, np.newaxis]
        detail1 = jwt.waverec2(detail1_coeffs, "haar")[:, :, :, np.newaxis]
        detail2 = jwt.waverec2(detail2_coeffs, "haar")[:, :, :, np.newaxis]

        approx_final = np.broadcast_to(approx, ϕ.coef.shape)
        detail1_final = np.broadcast_to(detail1, ϕ.coef.shape)
        detail2_final = np.broadcast_to(detail2, ϕ.coef.shape)

        approx_basis = T_approx.transform(approx_final)
        detail1_basis = T_detail1.transform(detail1_final)
        detail2_basis = T_detail2.transform(detail2_final)

        u_approx = approx_basis.to(*self.cfg["mode"])
        u_detail1 = detail1_basis.to(*self.cfg["mode"])
        u_detail2 = detail2_basis.to(*self.cfg["mode"])

        bias_approx = T_approx.transform(u_approx.grid(*u_approx.mode)).coef
        u_approx = u_approx.map(lambda coef: np.concatenate([coef, bias_approx], axis=-1))
        bias_detail1 = T_detail1.transform(u_detail1.grid(*u_detail1.mode)).coef
        u_detail1 = u_detail1.map(lambda coef: np.concatenate([coef, bias_detail1], axis=-1))
        bias_detail2 = T_detail2.transform(u_detail2.grid(*u_detail2.mode)).coef
        u_detail2 = u_detail2.map(lambda coef: np.concatenate([coef, bias_detail2], axis=-1))

        u_approx = u_approx.map(nn.Dense(self.cfg["hdim"] * 4))
        u_approx = T_approx.transform(self.activate(u_approx.inv()))
        u_detail1 = u_detail1.map(nn.Dense(self.cfg["hdim"] * 4))
        u_detail1 = T_detail1.transform(self.activate(u_detail1.inv()))
        u_detail2 = u_detail2.map(nn.Dense(self.cfg["hdim"] * 4))
        u_detail2 = T_detail2.transform(self.activate(u_detail2.inv()))

        u_approx = u_approx.map(nn.Dense(self.cfg["hdim"]))
        u_approx = T_approx.transform(self.activate(u_approx.inv()))
        u_detail1 = u_detail1.map(nn.Dense(self.cfg["hdim"]))
        u_detail1 = T_detail1.transform(self.activate(u_detail1.inv()))
        u_detail2 = u_detail2.map(nn.Dense(self.cfg["hdim"]))
        u_detail2 = T_detail2.transform(self.activate(u_detail2.inv()))

        for _ in range(self.cfg["depth"]):

            # process approx scale
            conv_approx = SpectralConv(self.cfg["hdim"])(u_approx)
            fc_approx = u_approx.map(nn.Dense(self.cfg["hdim"]))
            u_approx = T_approx.add(conv_approx, fc_approx)
            u_approx = T_approx.transform(self.activate(u_approx.inv()))
            
            # process detail1 scale
            conv_detail1 = SpectralConv(self.cfg["hdim"])(u_detail1)
            fc_detail1 = u_detail1.map(nn.Dense(self.cfg["hdim"]))
            u_detail1 = T_detail1.add(conv_detail1, fc_detail1)
            u_detail1 = T_detail1.transform(self.activate(u_detail1.inv()))
            
            # process detail2 scale
            conv_detail2 = SpectralConv(self.cfg["hdim"])(u_detail2)
            fc_detail2 = u_detail2.map(nn.Dense(self.cfg["hdim"]))
            u_detail2 = T_detail2.add(conv_detail2, fc_detail2)
            u_detail2 = T_detail2.transform(self.activate(u_detail2.inv()))

        u_approx = u_approx.map(nn.Dense(self.cfg["hdim"]))
        u_approx = T_approx.transform(self.activate(u_approx.inv()))
        u_detail1 = u_detail1.map(nn.Dense(self.cfg["hdim"]))
        u_detail1 = T_detail1.transform(self.activate(u_detail1.inv()))
        u_detail2 = u_detail2.map(nn.Dense(self.cfg["hdim"]))
        u_detail2 = T_detail2.transform(self.activate(u_detail2.inv()))

        u_approx = u_approx.map(nn.Dense(self.cfg["hdim"] * 4))
        u_approx = T_approx.transform(self.activate(u_approx.inv()))
        u_detail1 = u_detail1.map(nn.Dense(self.cfg["hdim"] * 4))
        u_detail1 = T_detail1.transform(self.activate(u_detail1.inv()))
        u_detail2 = u_detail2.map(nn.Dense(self.cfg["hdim"] * 4))
        u_detail2 = T_detail2.transform(self.activate(u_detail2.inv()))

        u_approx = u_approx.map(nn.Dense(self.pde.odim))
        u_detail1 = u_detail1.map(nn.Dense(self.pde.odim))
        u_detail2 = u_detail2.map(nn.Dense(self.pde.odim))  

        # recombine the different scales using wavelet reconstruction
        out_approx  = u_approx.inv()
        out_detail1 = u_detail1.inv()
        out_detail2 = u_detail2.inv()

        # reconstruct each batch item and output channel
        u_phys = out_approx + out_detail1 + out_detail2
        u = T_approx.transform(u_phys)
        return self.pde.mollifier(ϕ, u)
