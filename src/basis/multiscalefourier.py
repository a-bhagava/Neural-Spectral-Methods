from . import *
from .fourier import *

@struct.dataclass
class MultiScaleFourier(Fourier):
    """
    Multi-Scale Fourier: Fourier with offset band isolation.

    Stores full spectrum but zeros [0:offset) to isolate frequency band [offset:n/2].
    The `ix()` method returns offset-shifted indices so that truncation via `to()`
    operates on the correct frequency band.

    Use the MSF() factory function to create classes with specific offsets.
    """

    # Class-level offset (overridden by MSF factory)
    _offset: int = struct.field(pytree_node=False, default=0)

    @staticmethod
    def repr() -> str:
        return "M"
    
    @classmethod
    def fn(cls, n: int, x: X) -> X:
        offset = cls.get_offset()
        shifted_freqs = np.arange(n//2+1) + offset  # [offset, ..., offset+n//2]
        return np.moveaxis(real(np.moveaxis(np.exp(x * shifted_freqs * 2j * π), -1, 0), n), 0, -1)

    @classmethod
    def get_offset(cls) -> int:
        """Get the offset for this class. Override in subclasses."""
        return 0

    @classmethod
    def transform(cls, x: X):
        n = len(x)
        offset = cls.get_offset()

        coef = np.fft.rfft(x, axis=0, norm="forward")
        coef = coef.at[1:-(n//-2)].multiply(2)
        if offset > 0:
            coef = coef.at[:offset].set(0)

        real_coef = real(coef, n)
        return cls(real_coef)

    def inv(self) -> X:
        """
        Inverse transform using frequency modulation.

        For offset modes [offset:offset+n//2], we compute the inverse on a grid of
        size n by treating the coefficients as baseband [0:n//2] and modulating
        by e^{i * offset * x} where x is the grid.

        This is equivalent to: IFFT of full spectrum, but much more memory efficient.
        """
        n = len(self)  # truncated mode count (e.g., 31)
        offset = type(self).get_offset()
        coef = comp(self.coef, n)
        coef = coef.at[1:-(n // -2)].divide(2)
        x_baseband = np.fft.irfft(coef, n, axis=0, norm="forward")

        # Modulate by e^{i * offset * 2π * x} to shift back to true frequencies
        # Grid points: x_k = k/n for k = 0, ..., n-1
        grid = np.arange(n) / n
        modulation = np.exp(2j * π * offset * grid)
        modulation = np.expand_dims(modulation, axis=tuple(range(1, x_baseband.ndim)))
        return (x_baseband * modulation).real

    @classmethod
    def ix(cls, n: int) -> X:
        """
        Offset-aware index selection for truncation.

        Returns indices for frequencies [offset:offset+n) instead of [0:n).
        This makes truncation via `to()` work correctly with the offset band.

        The indices are symmetric around zero, then shifted:
        - Negative frequencies shift DOWN by offset (more negative)
        - Positive frequencies shift UP by offset

        Example:
            MSF(100).ix(31) returns indices for modes [-15-100, ..., -1-100, 0+100, ..., 15+100]
            i.e., [-115, ..., -101, 100, ..., 115]
        """
        offset = cls.get_offset()

        # Standard symmetric indices: [-n//2+1, ..., -1, 0, 1, ..., n//2]
        # Negative part: [-n//2+1, ..., -1] has (n//2 - 1) elements for even n, (n-1)//2 for odd
        # Positive part: [0, 1, ..., n//2] has (n//2 + 1) elements

        # Build directly without boolean indexing (JAX-compatible)
        neg_part = np.arange(-n // 2 + 1, 0)      # [-n//2+1, ..., -1]
        pos_part = np.arange(0, n // 2 + 1)       # [0, 1, ..., n//2]

        # Shift: negative frequencies go DOWN, positive frequencies go UP
        neg_shifted = neg_part - offset
        pos_shifted = pos_part + offset

        # Concatenate maintaining symmetric structure
        return np.concatenate([neg_shifted, pos_shifted])

    def grad(self, k: int = 1):
        """Gradient using offset-aware frequencies."""
        n = len(self)
        offset = type(self).get_offset()

        # Frequencies for offset band: [offset, offset+1, ..., offset+n//2]
        freqs = (np.arange(n//2+1) + offset) * 2j * π
        coef = np.expand_dims(freqs**k, range(1, self.coef.ndim))

        return type(self)(real(comp(self.coef, n) * coef, n)[..., None])


def MSF(mode_offset: int = 0):
    """
    Factory function to create MultiScaleFourier class with specific offset.

    Args:
        mode_offset: Starting frequency index for the band

    Returns:
        MultiScaleFourier class configured for frequencies [offset, offset+n)

    Example:
        MSF20 = MSF(20)  # Learn modes [20, 20+n)
        basis = MSF20.transform(signal)
    """
    @struct.dataclass
    class _MultiScaleFourier(MultiScaleFourier):

        @classmethod
        def get_offset(cls) -> int:
            return mode_offset

        @staticmethod
        def repr() -> str:
            return f"M{mode_offset}"

    return _MultiScaleFourier


# ---------------------------------------------------------------------------- #
#                                    HELPER                                    #
# ---------------------------------------------------------------------------- #

def real(coef: X, n: int) -> X:
    """Complex coef -> Real coef"""
    cos, sin = coef.real, coef.imag[1:-(n//-2)]
    return np.concatenate((cos, sin[::-1]), 0)


def comp(coef: X, n: int) -> X:
    """Real coef -> Complex coef"""
    cos, sin = np.split(coef, (m:=n//2+1, ))
    return (cos+0j).at[n-m:0:-1].add(sin*1j)