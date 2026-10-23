#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script investigating IMEX stability for advection-diffusion

.. literalinclude:: /../qmint/play/tibo/imexStabilityAdvDiffSDC.py
   :language: python
   :linenos:
   :lines: 11-
"""
import numpy as np

from qmint.utils.analysis import EigenvaluesIMEX, runDahlquistIMEX, IMEX_SCHEMES
from qmint.utils.plots import plotAdvDiffStability, plotAdvDiffMaxCFL

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
scheme = "SDC44"
params = IMEX_SCHEMES[scheme]

# -- space discretization
dim = 3
advOrder = 4
diffOrder = 4

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    eigs = EigenvaluesIMEX()
    (lamI, lamE), (cflAdv, meshPeclet) = \
        eigs.getAdvDiff(advOrder, diffOrder)

    stab, error, suffix = runDahlquistIMEX(lamI, lamE, scheme, **params)

    plotAdvDiffStability(
        stab, cflAdv, meshPeclet, error, figName=f"IMEX stability {suffix}")

    print("Computing maximum CFL vs Pe ...")
    idx = np.argmax(stab > 1, axis=0)
    cflMax = cflAdv[idx]
    print(" -- done !")

    plotAdvDiffMaxCFL(meshPeclet, cflMax, label=suffix)
