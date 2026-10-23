import numpy as np

from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDELTA_GENERATORS
from qmat.qcoeff.butcher import RK_SCHEMES

from qmint.steppers.dahlquist import DahlquistIMEX

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


IMEX_SCHEMES = {
    "RK443":
        {"schemeI": "ARK443ESDIRK", "paddingI": None,
         "schemeE": "ARK443ERK", "paddingE": None},
    "Euler":
        {"schemeI": "BE", "paddingI": "LEFT",
         "schemeE": "FE", "paddingE": "RIGHT"},
    "SDC44":
        dict(
            nNodes = 4,
            nSweeps = 4,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
        ),
    "SDC23":
        dict(
            nNodes = 2,
            nSweeps = 3,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
        ),
    }


def runDahlquistIMEX(lamI, lamE, scheme:str, stepUpdate=False, **params):

    problem = DahlquistIMEX(lamI, lamE)

    nPoints = max(np.prod(lamI.shape), np.prod(lamE.shape))

    if scheme.startswith("SDC"):
        nNodes, nSweeps, implSweep, explSweep, nodeType, quadType = \
            params.values()
        suffix = f"M{nNodes}_K{nSweeps}_{implSweep}_{explSweep}"

        print(f"Computing one SDC time-step on {nPoints} complex points ...")
        problem = DahlquistIMEX(lamI, lamE)

        coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)

        genI = QDELTA_GENERATORS[implSweep](qGen=coll)
        genE = QDELTA_GENERATORS[explSweep](qGen=coll)

        sweeps = [k+1 for k in range(nSweeps)]
        qDeltaI = genI.genCoeffs(k=sweeps)
        qDeltaE = genE.genCoeffs(k=sweeps)

        uNum = problem.runSDC(
            coll.Q, coll.weights if stepUpdate else None,
            qDeltaI, qDeltaE, nSweeps=nSweeps)

    else:
        print(f"Computing one IMEX RK time-step on {nPoints} complex points ...")
        suffix = f"{scheme}"

        schemeI, paddingI, schemeE, paddingE = params.values()

        schemeI = RK_SCHEMES[schemeI](padding=paddingI)
        schemeE = RK_SCHEMES[schemeE](padding=paddingE)

        QI = schemeI.Q
        wI = schemeI.weights if stepUpdate else None
        QE = schemeE.Q
        wE = schemeE.weights if stepUpdate else None

        uNum = problem.run(QI, wI, QE, wE)

    print(" -- done !")
    if stepUpdate:
        suffix += "_SU"

    u1 = uNum[-1]
    stab = np.abs(u1).max(axis=-1)
    error = np.abs(u1 - np.exp(lamI+lamE)).max(axis=-1)

    return stab, error, suffix
