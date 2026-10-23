#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:55:49 2026

@author: telu
"""
import numpy as np

from qmint.utils.analysis import EigenvaluesIMEX, runDahlquistIMEX, IMEX_SCHEMES
from qmint.utils.plots import plt

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
refScheme = "RK443"
compScheme = ["Euler", "SDC44", "SDC23"]


# -- space discretization
dim = 3
advOrder = np.inf
diffOrder = np.inf

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    eigs = EigenvaluesIMEX(nSamplesX=401, nWaveNumbers=16)
    (lamI, lamE), (cflAdv, meshPeclet) = eigs.getAdvDiff(
        advOrder, diffOrder, ePecletMin=0, cflMax=3)

    stabRef, _, _ = runDahlquistIMEX(
        lamI, lamE, refScheme, **IMEX_SCHEMES[refScheme])
    idx = np.argmax(stabRef > 1, axis=0)
    cflMaxRef = cflAdv[idx]


    results = {}
    for scheme in compScheme:
        stabComp, _, _ = runDahlquistIMEX(
            lamI, lamE, scheme, **IMEX_SCHEMES[scheme])
        idx = np.argmax(stabComp > 1, axis=0)
        results[scheme] = cflAdv[idx]

    plt.figure("time-step ratio")
    plt.clf()
    for scheme, cflMaxComp in results.items():
        plt.plot(meshPeclet, cflMaxComp/cflMaxRef, label=scheme)
    plt.ylabel(r"Max $\Delta{t}$ ratio VS "f"{refScheme}")
    plt.xlabel(r"$Pe_\Delta$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Comparison with RBC results
    plt.vlines(11, ymin=0, ymax=1.75, linestyles="--", colors="black")
    plt.text(13, 0.55, r"$Ra=1e^7$")

    plt.vlines(30, ymin=0, ymax=1.75, linestyles="--", colors="black")
    plt.text(32, 0.55, r"$Ra=1e^8$")
