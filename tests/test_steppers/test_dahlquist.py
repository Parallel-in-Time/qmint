import pytest
import numpy as np

from qmat import Q_GENERATORS, QDELTA_GENERATORS
from qmat.utils.sdc import solveDahlquistSDC

from qmint.steppers.dahlquist import Dahlquist, DahlquistIMEX

@pytest.mark.parametrize("lam", [1j, -1])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("nSteps", [1, 2, 5])
@pytest.mark.parametrize("tEnd", [1, 2, 5])
@pytest.mark.parametrize("scheme", ["RK4", "DIRK43", "Collocation"])
def testDahlquist(scheme, tEnd, nSteps, dim, lam):
    qGen = Q_GENERATORS[scheme].getInstance()

    lamVals = lam*np.linspace(0, 1, 4**dim).reshape((4,)*dim)
    ref = np.array([qGen.solveDahlquist(lam, 1, tEnd, nSteps)
                    for lam in lamVals.ravel()]).T.reshape((-1, *lamVals.shape))
    stepper = Dahlquist(lamVals, 1, tEnd, nSteps)

    sol1 = stepper.run(qGen.Q, qGen.weights)
    assert np.allclose(sol1, ref), \
        "Dahlquist stepper do not give the same solution as reference stepper"

    if scheme == "Collocation":
        assert np.allclose(qGen.nodes[-1], 1), \
            "default instance for Collocation does not have 1 as last node, but test depends on it"
        sol2 = stepper.run(qGen.Q, None)
        assert np.allclose(sol2, ref), \
            "Dahlquist without weights do not give the same solution as reference stepper"


@pytest.mark.parametrize("lam", [1j, -1])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("weights", [True, False])
@pytest.mark.parametrize("nSweeps", [1, 4])
@pytest.mark.parametrize("nSteps", [1, 5])
@pytest.mark.parametrize("tEnd", [1, 5])
@pytest.mark.parametrize("scheme", ["BE", "FE", "MIN-SR-FLEX"])
def testDahlquistSDC(scheme, tEnd, nSteps, nSweeps, weights, dim, lam):
    coll = Q_GENERATORS["Collocation"](nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    approx = QDELTA_GENERATORS[scheme](qGen=coll)
    nIters = [k+1 for k in range(nSweeps)] if scheme == "MIN-SR-FLEX" else nSweeps
    QDelta = approx.genCoeffs(k=nIters)

    lamVals = lam*np.linspace(0, 1, 4**dim).reshape((4,)*dim)
    ref = np.array([solveDahlquistSDC(lam, 1, tEnd, nSteps,
                                      nSweeps, coll.Q, QDelta, coll.weights if weights else None)
                    for lam in lamVals.ravel()]).T.reshape((-1, *lamVals.shape))

    stepper = Dahlquist(lamVals, 1, tEnd, nSteps)
    sol = stepper.runSDC(coll.Q, coll.weights if weights else None, QDelta, nSweeps)
    assert np.allclose(sol, ref), \
        "Dahlquist SDC stepper do not give the same solution as reference SDC stepper"


@pytest.mark.parametrize("lam", [1j, -1])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("nSteps", [1, 5])
@pytest.mark.parametrize("tEnd", [1, 5])
@pytest.mark.parametrize("scheme", ["RK4", "DIRK43", "Collocation"])
def testDahlquistIMEX(scheme, tEnd, nSteps, dim, lam):
    qGen = Q_GENERATORS[scheme].getInstance()

    lamVals = lam*np.linspace(0, 1, 4**dim).reshape((4,)*dim)

    basis = Dahlquist(lam=lamVals, u0=1, tEnd=tEnd, nSteps=nSteps)
    ref = basis.run(Q=qGen.Q, weights=qGen.weights)

    stepper = DahlquistIMEX(lamI=lamVals, lamE=[0], u0=1, tEnd=tEnd, nSteps=nSteps)
    sol = stepper.run(QI=qGen.Q, wI=qGen.weights, QE=qGen.Q, wE=qGen.weights)
    assert np.allclose(sol, ref), \
        "DahlquistIMEX stepper does not match Dahlquist stepper with implicit part only"

    if scheme in ["Collocation", "DIRK43"]:
        sol = stepper.run(QI=qGen.Q, wI=None, QE=qGen.Q, wE=None)
        assert np.allclose(sol, ref), \
            "DahlquistIMEX stepper without weights does not match Dahlquist stepper with implicit part only"

    stepper = DahlquistIMEX(lamI=[0], lamE=lamVals, u0=1, tEnd=tEnd, nSteps=nSteps)
    sol = stepper.run(QI=qGen.Q, wI=qGen.weights, QE=qGen.Q, wE=qGen.weights)
    assert np.allclose(sol, ref), \
        "DahlquistIMEX stepper does not match Dahlquist stepper with explicit part only"

    if scheme == "Collocation":
        sol = stepper.run(QI=qGen.Q, wI=None, QE=qGen.Q, wE=None)
        assert np.allclose(sol, ref), \
            "DahlquistIMEX stepper without weights does not match Dahlquist stepper with explicit part only"

    for weights in [qGen.weights, None]:
        basis = Dahlquist(lam=2*lamVals, u0=1, tEnd=tEnd, nSteps=nSteps)
        ref = basis.run(Q=qGen.Q, weights=weights)

        stepper = DahlquistIMEX(lamI=lamVals, lamE=lamVals, u0=1, tEnd=tEnd, nSteps=nSteps)
        sol = stepper.run(QI=qGen.Q, wI=weights, QE=qGen.Q, wE=weights)
        detail = " with weights " if weights is not None else ""
        assert np.allclose(sol, ref), \
            f"DahlquistIMEX stepper {detail} does not produce the linear combination of IMEX sum"


@pytest.mark.parametrize("lam", [1j, -1])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("weights", [True, False])
@pytest.mark.parametrize("nSweeps", [1, 4])
@pytest.mark.parametrize("nSteps", [1, 5])
@pytest.mark.parametrize("tEnd", [1, 5])
@pytest.mark.parametrize("scheme", ["BE", "FE", "MIN-SR-FLEX"])
def testDahlquistIMEXSDC(scheme, tEnd, nSteps, nSweeps, weights, dim, lam):
    coll = Q_GENERATORS["Collocation"](nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    approx = QDELTA_GENERATORS[scheme](qGen=coll)
    nIters = [k+1 for k in range(nSweeps)] if scheme == "MIN-SR-FLEX" else nSweeps
    QDelta = approx.genCoeffs(k=nIters)

    lamVals = lam*np.linspace(0, 1, 4**dim).reshape((4,)*dim)
    ref = np.array([solveDahlquistSDC(lam, 1, tEnd, nSteps,
                                      nSweeps, coll.Q, QDelta, coll.weights if weights else None)
                    for lam in lamVals.ravel()]).T.reshape((-1, *lamVals.shape))

    stepper = DahlquistIMEX(lamVals, [0], 1, tEnd, nSteps)
    sol = stepper.runSDC(coll.Q, coll.weights if weights else None, QDelta, QDelta, nSweeps)
    assert np.allclose(sol, ref), \
        "DahlquistIMEX SDC stepper does not match reference stepper for implicit only"

    stepper = DahlquistIMEX([0], lamVals, 1, tEnd, nSteps)
    sol = stepper.runSDC(coll.Q, coll.weights if weights else None, QDelta, QDelta, nSweeps)
    assert np.allclose(sol, ref), \
        "DahlquistIMEX SDC stepper does not match reference stepper for explicit only"

    stepper = DahlquistIMEX(lamVals, lamVals, 1, tEnd, nSteps)
    sol = stepper.runSDC(coll.Q, coll.weights if weights else None, QDelta, QDelta, nSweeps)
    ref = np.array([solveDahlquistSDC(2*lam, 1, tEnd, nSteps,
                                      nSweeps, coll.Q, QDelta, coll.weights if weights else None)
                    for lam in lamVals.ravel()]).T.reshape((-1, *lamVals.shape))
    assert np.allclose(sol, ref), \
        "DahlquistIMEX SDC stepper does not match reference stepper with IMEX sum"
