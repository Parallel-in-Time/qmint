import numpy as np
import matplotlib.pyplot as plt

# General matplotlib settings
plt.rc('font', size=12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams['lines.markersize'] = 7.0
plt.rcParams['lines.markeredgewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['figure.max_open_warning'] = 100


def plotAdvDiffStability(stab, cflAdv, meshPeclet, error=None, figName=None, cleanFig=True):
    plt.figure(figName)
    if cleanFig:
        plt.clf()

    coords = (meshPeclet, cflAdv)
    plt.contourf(*coords, stab, levels=[0, 1, np.inf], colors=['white', 'gainsboro'])
    plt.contour(*coords, stab, levels=[1], colors="black")
    if error is not None:
        plt.contour(*coords, error, levels=[1e-1], colors="black", linestyles=":")
        plt.contour(*coords, error, levels=[1e-2], colors="black", linestyles="--")
    plt.grid(True)
    plt.xscale('log')
    plt.ylabel(r"$\sigma_{adv}$")
    plt.xlabel(r"$Pe_\Delta$")
    plt.tight_layout()


def plotAdvDiffMaxCFL(meshPeclet, cflMax, label, figName="CFL max"):
    plt.figure(figName)
    plt.semilogx(meshPeclet, cflMax, label=label)
    plt.ylabel(r"$\sigma_{adv,max}$")
    plt.xlabel(r"$Pe_\Delta$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
