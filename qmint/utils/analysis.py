import numpy as np


ADV_COEFFS = {
    1: ([0, 1, -1], 1),
    2: ([1, 0, -1], 2),
    3: ([0, 2, 3, -6, 1], 6),
    4: ([-1, 8, 0, -8, 1], 12),
    5: ([0, -3, 30, 20, -60, 15, -2], 60),
    6: ([1, -9, 45, 0, -45, 9, -1], 60)
    }

DIFF_COEFFS = {
    2: ([1, -2, 1], 1),
    4: ([-1, 16, -30, 16, -1], 12),
    6: ([2, -27, 270, -490, 270, -27, 2], 180),
    }

def getFourierSymbol(deriv, order):

    # Exact Fourier symbols
    if order == np.inf:
        if deriv == 1:  # advection
            def z(theta):
                return 1j*theta
        elif deriv == 2: # diffusion
            def z(theta):
                return -theta**2 * (1 + 0j)
        else:
            raise NotImplementedError(f"{deriv=}")

        return z

    # Discrete Fourier symbol
    table = {
        1: ADV_COEFFS,
        2: DIFF_COEFFS,
    }[deriv]

    def z(theta):
        coeffs, div = table[order]
        s = len(coeffs)
        exponents = range((s//2), -(s//2)-1, -1)
        symbol = sum(c*np.exp(1j*theta*e) for c, e in zip(coeffs, exponents))
        symbol /= div
        symbol = (1 + 0j)*symbol
        return symbol

    return z


class EigenvaluesIMEX():

    def __init__(self, nSamplesX=256, nWaveNumbers=32, nSamplesY=None):
        self.nSamplesX = nSamplesX
        self.nSamplesY = nSamplesX if nSamplesY is None else nSamplesY
        self.nWaveNumbers = nWaveNumbers

    def getAdvDiff(self, advOrder, diffOrder, dim=3, cflMax=5, ePecletMin=-1, ePecletMax=2):
        theta = np.linspace(0, np.pi, num=self.nWaveNumbers+1)[1:]
        cflAdv = np.linspace(0, cflMax, num=self.nSamplesY)
        meshPeclet = np.logspace(ePecletMin, ePecletMax, num=self.nSamplesX)

        zAdv = getFourierSymbol(1, advOrder)
        zDiff = getFourierSymbol(2, diffOrder)

        lamE = -dim*cflAdv[:, None, None]*zAdv(theta)[None, None, :]
        lamI = dim*cflAdv[:, None, None]/meshPeclet[None, :, None]*zDiff(theta)[None, None, :]

        return (lamI, lamE), (cflAdv, meshPeclet)
