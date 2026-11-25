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

# Lele compact schemes (JCP 103, 1992): (alpha, a, b) per order
ADV_COMPACT = {
    2: (0.0, 1.0, 0.0),
    4: (0.25, 1.5, 0.0),
    6: (1/3, 14/9, 1/9),
    }
DIFF_COMPACT = {
    2: (0.0, 1.0, 0.0),
    4: (0.1, 1.2, 0.0),
    6: (2/11, 12/11, 3/11),
    }


def getFourierSymbol(deriv, order, spaceScheme="FD"):

    # Spectral discretization (exact Fourier symbols)
    if order in [np.inf, -1]:
        if deriv == 1:  # advection
            def z(theta):
                return 1j*theta
        elif deriv == 2: # diffusion
            def z(theta):
                return -theta**2 * (1 + 0j)
        else:
            raise NotImplementedError(f"{deriv=}")

        return z

    # Lele compact schemes: rational symbols
    #   1st deriv: z = i (a sin t + (b/2) sin 2t) / (1 + 2 alpha cos t)
    #   2nd deriv: z = (2a (cos t - 1) + (b/2) (cos 2t - 1)) / (1 + 2 alpha cos t)
    if spaceScheme.upper() in ("COMPACT", "LELE"):
        table = {1: ADV_COMPACT, 2: DIFF_COMPACT}[deriv]
        alpha, a, b = table[order]

        def zCompact(theta):
            theta = np.asarray(theta)
            denom = 1 + 2*alpha*np.cos(theta)
            if deriv == 1:
                return 1j*(a*np.sin(theta) + 0.5*b*np.sin(2*theta))/denom
            else:
                return (2*a*(np.cos(theta)-1)
                        + 0.5*b*(np.cos(2*theta)-1))/denom + 0j

        return zCompact

    # Finite differences: polynomial symbols
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

    def getAdvDiff(self, advOrder, diffOrder, dim=3, cflMax=5, ePecletMin=-1, ePecletMax=2,
                   spaceScheme="FD", returnExact=False):
        theta = np.linspace(0, np.pi, num=self.nWaveNumbers+1)[1:]
        cflAdv = np.linspace(0, cflMax, num=self.nSamplesY+1)[1:]
        meshPeclet = np.logspace(ePecletMin, ePecletMax, num=self.nSamplesX)

        zAdv = getFourierSymbol(1, advOrder, spaceScheme)
        zDiff = getFourierSymbol(2, diffOrder, spaceScheme)

        lamE = -dim*cflAdv[:, None, None]*zAdv(theta)[None, None, :]
        lamI = dim*cflAdv[:, None, None]/meshPeclet[None, :, None]*zDiff(theta)[None, None, :]

        if returnExact:
            # exact PDE eigenvalues (exact Fourier symbols i*theta and -theta^2),
            # used as reference for the space-time error
            zAdvEx = getFourierSymbol(1, np.inf)
            zDiffEx = getFourierSymbol(2, np.inf)
            lamE_ex = -dim*cflAdv[:, None, None]*zAdvEx(theta)[None, None, :]
            lamI_ex = dim*cflAdv[:, None, None]/meshPeclet[None, :, None]*zDiffEx(theta)[None, None, :]
            return (lamI, lamE), (cflAdv, meshPeclet), (lamI_ex, lamE_ex)

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
            theta=0,
        ),
    "SDC23":
        dict(
            nNodes = 2,
            nSweeps = 3,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta=0,
        ),
    "SDC44-T":
        dict(
            nNodes = 4,
            nSweeps = 4,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta=0.5,
        ),
    # Stiller's original semi-implicit SDC (SDC-SI): classical marching sweeps
    # with BE (implicit) / FE (explicit) preconditioners, plus the Lax--Wendroff
    # term (theta_m/2) lamE^2 with theta_m = Delta t_m (= diag(QDeltaI) in units
    # of dt) on BOTH sides of each node equation (runSDC theta="stiller").
    # Fixed point = collocation solution (order preserved).
    "SDC23-ST":
        dict(
            nNodes = 2,
            nSweeps = 3,
            implSweep = "BE",
            explSweep = "FE",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta="stiller",
        ),
    "SDC44-ST":
        dict(
            nNodes = 4,
            nSweeps = 4,
            implSweep = "BE",
            explSweep = "FE",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta="stiller",
        ),
    # Stiller's original semi-implicit SDC (SDC-SI), via runSDCStiller:
    # classical marching sweeps + LW term (theta_m/2) lamE^2 with theta_m=Delta t_m
    # on BOTH sides of each node equation.  stillerSI = (sPredictor, sCorrector)
    # stages of the SI1 integrators.  SDC-SI(1,1)^3_2 is L-stable (order 3);
    # SDC-SI(2,2)^15_6 is L-stable up to order 11 (Stiller, Table 1).
    "SDC23-S":
        dict(
            nNodes = 2,
            nSweeps = 3,
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            stillerSI=(1, 1),
        ),
    "SDC26-S":
        dict(
            nNodes = 6,
            nSweeps = 15,
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            stillerSI=(2, 2),
        ),
    # Stiller's Table 1 order-7 configuration: M=4 Radau points, K=8 sweeps
    # with the two-stage SI1(2) predictor/corrector -- A-/L-stable
    "SDC44-S":
        dict(
            nNodes = 4,
            nSweeps = 8,
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            stillerSI=(2, 2),
        ),
    # Marching sweeps with the both-sides S-term: fraction sAdv of the advection
    # moved into the implicit preconditioner (runSDC sAdv parameter)
    "SDC23-SA":
        dict(
            nNodes = 2,
            nSweeps = 3,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta=0,
            sAdv=2.0,
        ),
    "SDC44-SA":
        dict(
            nNodes = 4,
            nSweeps = 4,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
            theta=0,
            sAdv=2.0,
        ),
    # Parallel stabilized PIMEX-SDC: diagonal QDelta (MIN-SR-S, as in the plain
    # SDC schemes above) + Stiller's theta-term (theta/2) A_c^2 added on BOTH
    # sides of each node's own diagonal implicit solve (theta_m = q_m * dt),
    # so the node updates stay decoupled (parallel) and the fixed point stays
    # the collocation solution (order preserved).
    "SDC23-PS":
        dict(nNodes=2, nSweeps=3, parstiller=True),
    "SDC44-PS":
        dict(nNodes=4, nSweeps=4, parstiller=True),
    # Parallel S-term PIMEX-SDC: same diagonal QDelta sweeps, but the node-local
    # term added on BOTH sides of each node's diagonal solve is a fraction
    # sAdv of the advection itself (frozen Jacobian), i.e. per mode
    # zI -> zI + sAdv * q_m * zE.  Fixed point stays the collocation solution
    # (order preserved), node updates stay decoupled (parallel).
    "SDC23-PA":
        dict(nNodes=2, nSweeps=3, parstiller=True, sAdv=2.0, thetaScale=0.0),
    "SDC44-PA":
        dict(nNodes=4, nSweeps=4, parstiller=True, sAdv=2.0, thetaScale=0.0),
    }


def runDahlquistIMEXParStiller(lamI, lamE, nNodes=4, nSweeps=4,
                               nodeType="LEGENDRE", quadType="RADAU-RIGHT",
                               thetaScale=1.0, sAdv=0.0):
    r"""Parallel stabilized PIMEX-SDC: diagonal QDelta sweeps (MIN-SR-S) with
    Stiller's theta-term added to each node's own implicit solve.

    Sweep k updates every node m independently (parallel across nodes):

    .. math::
        u_m^{k+1} = u_0 + S_m^k + \Delta{t} Q_\Delta[m,m] [
            \phi_{I}(u_m^{k+1}) - \phi_{I}(u_m^k)
            ],

    with

    .. math::
        S_m^k = \Delta{t} \sum_{i=0}^{M-1} Q[m,i] (\lambda_I + \lambda_E) u_i^k

    the Q evaluation on the old sweep, available at all nodes simultaneously and

    .. math::
        \phi_I = (\lambda_I - \vartheta * Q_\Delta[m,m]/2 * Im(lamE)^2) u,
        theta_m = q_m * dt.

    The theta-term is local to node m, so the block-diagonal structure of the
    parallel sweeps is preserved.  It cancels at the fixed point (collocation
    solution, order preserved) while damping the sweep iteration.

    With sAdv > 0 (S-term, SDC-PA), a fraction sAdv of the advection itself is
    additionally moved into the node-local implicit solve on BOTH sides:
    per mode zI -> zI + sAdv * q_m * zE (in addition to the theta-term).

    Returns
    -------
    uLast : array of shape lamI.shape
        Last-node solution after `nSweeps` sweeps.
    """
    from qmat.qdelta.diag import MIN_SR_S
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    Q = coll.Q
    Qd = np.asarray(MIN_SR_S(qGen=coll).genCoeffs(k=list(range(1, nSweeps+1))))
    lam = lamI + lamE
    zi2 = lamE.imag**2
    one = np.ones(lamI.shape, dtype=complex)
    u = np.ones((nNodes, *lamI.shape), dtype=complex)  # copy initialization
    for k in range(nSweeps):
        qm = Qd[k].diagonal()
        uNew = np.zeros_like(u)
        for m in range(nNodes):
            zI = lamI + sAdv * qm[m] * lamE - thetaScale * qm[m]/2 * zi2
            Sm = np.einsum('i,i...->...', Q[m], lam * u)
            uNew[m] = (one + Sm - qm[m]*zI*u[m]) / (1 - qm[m]*zI)
        u = uNew
    return u[-1]


def runDahlquistIMEX(lamI, lamE, scheme:str, stepUpdate=False,
                     errorRef=None, errorMask=None, **params):

    problem = DahlquistIMEX(lamI, lamE)

    nPoints = max(np.prod(lamI.shape), np.prod(lamE.shape))

    if scheme.startswith("SDC"):
        nNodes = params["nNodes"]
        nSweeps = params["nSweeps"]
        if params.get("parstiller", False):
            # parallel stabilized PIMEX-SDC: diagonal QDelta + node-local
            # Stiller theta-term (both sides) and/or S-term sAdv (both sides)
            sAdv = params.get("sAdv", 0.0)
            suffix = f"M{nNodes}_K{nSweeps}_PARSTILLER" + (f"_S{sAdv}" if sAdv else "")
            u1 = runDahlquistIMEXParStiller(lamI, lamE, nNodes=nNodes, nSweeps=nSweeps,
                                            thetaScale=params.get("thetaScale", 1.0),
                                            sAdv=sAdv)
            stab = np.abs(u1).max(axis=-1)
            if errorRef is None:
                errFull = np.abs(u1 - np.exp(lamI+lamE))
            else:
                errFull = np.abs(u1 - np.exp(errorRef[0]+errorRef[1]))
            if errorMask is not None:
                errFull = errFull[..., errorMask]
            error = errFull.max(axis=-1)
            return stab, error, suffix
        nodeType = params["nodeType"]
        quadType = params["quadType"]

        print(f"Computing one SDC time-step on {nPoints} complex points ...")
        problem = DahlquistIMEX(lamI, lamE)

        coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)

        if "stillerSI" in params:
            # Stiller's original SDC-SI: dedicated marching stepper
            # (runSDCStiller), classical subinterval preconditioner + both-sides
            # LW term theta_m = Delta t_m
            sP, sC = params["stillerSI"]
            uNum = problem.runSDCStiller(
                coll.Q, nSweeps, sPredictor=sP, sCorrector=sC,
                weights=coll.weights if stepUpdate else None)
            suffix = f"M{nNodes}_K{nSweeps}_SI{sP}{sC}"
            u1 = uNum[-1]
            stab = np.abs(u1).max(axis=-1)
            if errorRef is None:
                errFull = np.abs(u1 - np.exp(lamI+lamE))
            else:
                errFull = np.abs(u1 - np.exp(errorRef[0]+errorRef[1]))
            if errorMask is not None:
                errFull = errFull[..., errorMask]
            error = errFull.max(axis=-1)
            print(" -- done !")
            return stab, error, suffix

        implSweep = params["implSweep"]
        explSweep = params["explSweep"]
        theta = params.get("theta", 0)
        epsilon = params.get("epsilon", 0)
        sAdv = params.get("sAdv", 0)
        suffix = f"M{nNodes}_K{nSweeps}_{implSweep}_{explSweep}" + (f"_S{sAdv}" if sAdv else "")

        genI = QDELTA_GENERATORS[implSweep](qGen=coll)
        genE = QDELTA_GENERATORS[explSweep](qGen=coll)

        sweeps = [k+1 for k in range(nSweeps)]
        qDeltaI = genI.genCoeffs(k=sweeps)
        qDeltaE = genE.genCoeffs(k=sweeps)

        if isinstance(theta, str) and theta == "stiller":
            # Stiller's original choice theta_m = Delta t_m = q_m * dt
            theta = np.diagonal(qDeltaI[0])

        uNum = problem.runSDC(
            coll.Q, coll.weights if stepUpdate else None,
            qDeltaI, qDeltaE, nSweeps=nSweeps, theta=theta, epsilon=epsilon,
            sAdv=sAdv)

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
    if errorRef is None:
        # temporal error w.r.t. the semi-discrete (discrete symbols) solution
        errFull = np.abs(u1 - np.exp(lamI+lamE))
    else:
        # space-time error w.r.t. the exact PDE solution
        lamI_ex, lamE_ex = errorRef
        errFull = np.abs(u1 - np.exp(lamI_ex+lamE_ex))
    if errorMask is not None:
        # restrict the accuracy metric to selected wave numbers (last axis),
        # e.g. only well-resolved modes theta <= pi/2
        errFull = errFull[..., errorMask]
    error = errFull.max(axis=-1)

    return stab, error, suffix


def getMaxCFL(stab, cflAdv):
    n = len(cflAdv)
    idx = np.argmax(stab > 1, axis=0)
    # if never unstable, argmax returns 0 but the scheme is stable up to cflMax:
    # detect via the last sample and cap at cflAdv[-1]
    neverUnstable = stab[-1] <= 1
    idx = np.minimum(idx, n-2)

    x1 = cflAdv[idx]
    x2 = cflAdv[idx+1]

    y1 = stab[idx, range(len(idx))]
    y2 = stab[idx+1, range(len(idx))]

    # y_i = a x_i + b
    a = (y2-y1)/(x2-x1)
    b = (y1*(x2-x1) - x1*(y2-y1))/(x2-x1)
    # 1 = a x_lim + b
    cflMax = (1-b)/a

    # schemes stable on the whole sampled range are capped at cflAdv[-1];
    # schemes unstable from the first sample give x_lim < 0, clipped to 0
    cflMax = np.where(neverUnstable, cflAdv[-1], cflMax)
    cflMax = np.clip(cflMax, 0, cflAdv[-1])

    return cflMax
