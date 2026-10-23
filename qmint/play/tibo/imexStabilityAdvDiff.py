#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script investigating IMEX stability for advection-diffusion solved with SDC

.. literalinclude:: /../qmint/play/tibo/imexStabilityAdvDiffSDC.py
   :language: python
   :linenos:
   :lines: 11-
"""
import numpy as np

from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDELTA_GENERATORS
from qmat.qcoeff.butcher import RK_SCHEMES

from qmint.utils.analysis import EigenvaluesIMEX
from qmint.steppers.dahlquist import DahlquistIMEX
from qmint.utils.plots import plotAdvDiffStability, plotAdvDiffMaxCFL

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
scheme = "SDC"
params = {
    "RK443":
        {"schemeI": "ARK443ESDIRK", "paddingI": None,
         "schemeE": "ARK443ERK", "paddingE": None},
    "Euler":
        {"schemeI": "BE", "paddingI": "LEFT",
         "schemeE": "FE", "paddingE": "RIGHT"},
    "SDC":
        dict(
            nNodes = 4,
            nSweeps = 4,
            implSweep = "MIN-SR-S",
            explSweep = "PIC",
            nodeType="LEGENDRE",
            quadType="RADAU-RIGHT",
        )
    }[scheme]

# -- space discretization
dim = 3
advOrder = 4
diffOrder = 4

eigs = EigenvaluesIMEX()
(lamI, lamE), (cflAdv, meshPeclet) = eigs.getAdvDiff(advOrder, diffOrder)


def runDahlquistIMEX(lamI, lamE, scheme, stepUpdate=False, **params):

    problem = DahlquistIMEX(lamI, lamE)

    if scheme == "SDC":
        nNodes, nSweeps, implSweep, explSweep, nodeType, quadType = \
            params.values()
        suffix = f"M{nNodes}_K{nSweeps}_{implSweep}_{explSweep}"

        print(f"Computing one SDC time-step on {lamI.size} complex points ...")
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
        print(f"Computing one IMEX RK time-step on {lamI.size} complex points ...")
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


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    stab, error, suffix = runDahlquistIMEX(lamI, lamE, scheme, **params)

    plotAdvDiffStability(
        stab, cflAdv, meshPeclet, error, figName=f"IMEX stability {suffix}")

    print("Computing maximum CFL vs Pe ...")
    idx = np.argmax(stab > 1, axis=0)
    cflMax = cflAdv[idx]
    print(" -- done !")

    plotAdvDiffMaxCFL(meshPeclet, cflMax, label=suffix)
