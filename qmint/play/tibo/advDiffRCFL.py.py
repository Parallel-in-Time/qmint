#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:55:49 2026

@author: telu
"""
from qmint.utils.analysis import \
    EigenvaluesIMEX, runDahlquistIMEX, IMEX_SCHEMES, getMaxCFL
from qmint.utils.plots import plt

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
refScheme = "RK443"
compScheme = ["Euler", "SDC44", "SDC23"]


# -- space discretization
dim = 3
advOrder = -1
diffOrder = -1

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    eigs = EigenvaluesIMEX(nSamplesX=401, nWaveNumbers=16)
    (lamI, lamE), (cflAdv, meshPeclet) = eigs.getAdvDiff(
        advOrder, diffOrder, ePecletMin=0, ePecletMax=4, cflMax=3)

    stabRef, _, _ = runDahlquistIMEX(
        lamI, lamE, refScheme, **IMEX_SCHEMES[refScheme])

    cflMaxRef = getMaxCFL(stabRef, cflAdv)

    results = {}
    for scheme in compScheme:
        stabComp, _, _ = runDahlquistIMEX(
            lamI, lamE, scheme, **IMEX_SCHEMES[scheme])

        cflMaxComp = getMaxCFL(stabComp, cflAdv)
        results[scheme] = cflMaxComp

    plt.figure("time-step ratio")
    plt.clf()
    for scheme, cflMaxComp in results.items():
        plt.semilogx(meshPeclet, cflMaxComp/cflMaxRef, label=scheme)
    plt.ylabel(r"Max $\Delta{t}$ ratio VS "f"{refScheme}")
    plt.xlabel(r"$Pe_\Delta$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Comparison with RBC results
    plt.vlines(11, ymin=0, ymax=1.75, linestyles="--", colors="black")
    plt.text(13, 0.55, r"$Ra=1e^7$", fontsize=16)

    plt.vlines(30, ymin=0, ymax=1.75, linestyles="--", colors="black")
    plt.text(36, 0.35, r"$Ra=1e^8$", fontsize=16)
