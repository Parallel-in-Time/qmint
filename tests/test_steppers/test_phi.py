import pytest
import numpy as np

from qmat import Q_GENERATORS, QDELTA_GENERATORS

from qmint.steppers.coeff import CoeffStepper
from qmint.steppers.phi import PhiStepper
from qmint.steppers.phi.euler import ForwardEuler, BackwardEuler
from qmint.diffops import DIFFOPS

EQUIVALENCES: dict[str, type[PhiStepper]] = {
    "FE": ForwardEuler,
    "BE": BackwardEuler,
}


@pytest.mark.parametrize("nNodes", [1, 4, 10])
@pytest.mark.parametrize("problem", ["Lorenz", "ProtheroRobinson"])
@pytest.mark.parametrize("scheme", EQUIVALENCES.keys())
def testPhiStepper(scheme, problem, nNodes):
    diffOp = DIFFOPS[problem]()
    tEnd = 0.1
    nSteps = 10*nNodes

    qGen = Q_GENERATORS[scheme].getInstance()

    refStepper = CoeffStepper(diffOp, tEnd=tEnd, nSteps=nSteps)
    ref = refStepper.run(qGen.Q, qGen.weights)

    regNodes = np.linspace(0, 1, num=nNodes+1)[1:]

    phiStepper = EQUIVALENCES[scheme](diffOp, nodes=regNodes, tEnd=tEnd, nSteps=nSteps//nNodes)
    sol = phiStepper.run()

    assert np.allclose(sol, ref[::nNodes]), \
        f"{phiStepper.__class__.__name__}-PhiStepper does not match equivalent CoeffStepper result"


@pytest.mark.parametrize("nSweeps", [1, 2, 4])
@pytest.mark.parametrize("quadType", ["RADAU-RIGHT", "LOBATTO"])
@pytest.mark.parametrize("nNodes", [2, 4, 8])
@pytest.mark.parametrize("problem", ["Lorenz", "ProtheroRobinson"])
@pytest.mark.parametrize("scheme", EQUIVALENCES.keys())
def testPhiSolverSDC(scheme, problem, nNodes, quadType, nSweeps):
    pParams = {}
    if problem == "ProtheroRobinson":
        pParams = {"epsilon": 0.01, "nonLinear": True}

    diffOp = DIFFOPS[problem](**pParams)
    tEnd = 0.1
    nSteps = 10

    coll = Q_GENERATORS["Collocation"](nNodes=nNodes, quadType=quadType, nodeType="LEGENDRE")
    approx = QDELTA_GENERATORS[scheme](qGen=coll)

    refStepper = CoeffStepper(diffOp, tEnd=tEnd, nSteps=nSteps)
    ref = refStepper.runSDC(nSweeps, coll.Q, coll.weights, approx.getQDelta())

    phiStepper = EQUIVALENCES[scheme](diffOp, nodes=coll.nodes, tEnd=tEnd, nSteps=nSteps)

    sol = phiStepper.runSDC(nSweeps, Q=coll.Q, weights=True)
    assert np.allclose(sol, ref), \
        f"{phiStepper.__class__.__name__}-PhiStepper SDC with given Q does not match equivalent CoeffStepper SDC result"

    sol = phiStepper.runSDC(nSweeps, Q=None, weights=True)
    assert np.allclose(sol, ref), \
        f"{phiStepper.__class__.__name__}-PhiStepper SDC does not match equivalent CoeffStepper SDC result"

    ref = refStepper.runSDC(nSweeps, coll.Q, None, approx.getQDelta())
    sol = phiStepper.runSDC(nSweeps, weights=None)
    assert np.allclose(sol, ref), \
        f"{phiStepper.__class__.__name__}-PhiStepper SDC without weights does not match equivalent CoeffStepper SDC result"

    if scheme == "BE":
        original = BackwardEuler.phiSolve
        BackwardEuler.phiSolve = PhiStepper.phiSolve  # use default phiSolve
        sol = phiStepper.runSDC(nSweeps, Q=coll.Q, weights=False)
        BackwardEuler.phiSolve = original
        assert np.allclose(sol, ref), \
            f"{phiStepper.__class__.__name__}-PhiStepper SDC with default phiSolve does not match equivalent CoeffStepper SDC result"
