import numpy as np

from typing import TypeVar
from qmat.utils import checkOverriding, storeClass, importAll
from qmat.lagrange import LagrangeApproximation

from qmint.steppers.coeff import CoeffStepper, DiffOp
import qmint.utils as utils


class PhiStepper(CoeffStepper):
    r"""
    Time-stepper for generic (non-linear) ODE system using :math:`\phi`-based schemes.

    It considers the following ODE :

    .. math::
        \frac{du}{dt} = f(u,t), \quad u(0)=u_0,

    and compute for each step the solution on **time nodes** :math:`\tau_1, ..., \tau_M`
    by solving the following system :

    .. math::

        u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}) = u_0.

    It uses then per default the last node solution :math:`u_{M}` as initial
    solution for the next step.

    ⚙️ Requires the implementation of an `evalPhi` method that evaluates
    the :math:`\phi` function.
    Also, a default `phiSolve` method is implemented, that solves
    the system above, and can be overridden for specific time-integrator
    (in particular for explicit time-integrators).
    Finally, it implements a default `stepUpdate` method that setup the
    next time-step using the last time-node solution.

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    nodes : 1D array-like
        The time nodes :math:`\tau_1, ..., \tau_M`.
    tEnd : float, optional
        Final simulation time :math:`T_{end}`. The default is 1.
    nSteps : int, optional
        Number :math:`N` of time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time :math:`t_0`. The default is 0.
    """
    def __init__(self, diffOp:DiffOp, nodes, tEnd=1, nSteps=1, t0=0):
        super().__init__(diffOp, tEnd, nSteps, t0)
        self.nodes = np.asarray(nodes, dtype=float)
        """Time nodes for each time-step of the time-integrator."""

    @property
    def nNodes(self):
        """Number of time-nodes"""
        return self.nodes.size


    def evalPhi(self, uVals, fEvals, out, t0=0):
        r"""
        Evaluate the :math:`\phi` operator on time-node up to :math:`u_{m+1}`.

        Considering :math:`u_0, u_1, \dots, u_{m+1}`,
        if evaluates :

        .. math::

            \phi(u_0, u_1, ..., u_{m}, u_{m+1}),

        and store its value into the output vector `out`.
        It also takes the node evaluation
        :math:`f(u_0,t_0),f(u_1,\tau_1),...,f(u_{m},\tau_{m})`
        as arguments, in order to avoid any additional :math:`f(u,t)`
        evaluations.

        Parameters
        ----------
        uVals : list[np.ndarray] of size :math:`m+2`
            The :math:`m+1` time-node solutions + the initial solution :math:`u_0`.
        fEvals : list[np.ndarray] of size :math:`m+1` or :math:`m+1`
            The :math:`f(u,t)` evaluations at each time nodes (+ initial solution),
            up to time-node :math:`m`.
            It can eventually contain a pre-computed :math:`f_{m+1}`
            to spare one :math:`f(u,t)` evaluation.
        out : np.ndarray
            Array used to store the evaluation.
        t0 : float, optional
            Initial step time. The default is 0.
        """
        raise NotImplementedError(
            "specialized PhiSolver must implement its evalPhi method")


    def phiSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        r"""
        Solve the node update at given time-node :math:`\tau_{m+1}`.

        Considering :math:`m+1` previous known node solutions
        :math:`u_0, u_1, ..., u_{m}`, it solves the following system :

        .. math::

            u -\phi(u_0, u_1, ..., u_{m}, u)
            = rhs,

        where the value given in `out` is used as **initial guess** and
        to **store the computed solution**.
        It also takes as argument the :math:`f` evaluations
        :math:`f_0, f_1, ..., f_{m}` to avoid supplementar re-computing those.

        Parameters
        ----------
        uPrev : list[np.ndarray] of size :math:`m+1`
            The previous node solutions :math:`u_0, u_1, ..., u_{m}`.
        fEvals : list[np.ndarray] of size :math:`m+1`
            Evaluations of previous node solutions :math:`f_0, f_1, ..., f_{m}`.
        out : np.ndarray
            Array with the initial guess, used to store the final solution.
        rhs : np.ndarray or float, optional
            Right hand side used to solve the equation above.
            The default is 0.
        t0 : float, optional
            Initial step size. The default is 0.
        """
        assert len(fEvals) == len(uPrev)

        def func(u:np.ndarray):
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalPhi([*uPrev, u], fEvals, out=res, t0=t0)
            res *= -1
            res += u
            res -= rhs
            return res.ravel()

        sol = self.diffOp.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    def stepUpdate(self, u0, uNodes, fEvals, out):
        r"""
        Update end-step solution to be used as initial guess for next step.

        Note
        ----
        This method has to ensures that fEvals[0] contains the :math:`f(u,t)`
        evaluation of the next step initial solution.

        Parameters
        ----------
        u0 : np.ndarray
            Initial solution for the current step.
        uNodes : list[np.ndarray]
            Precomputed node solutions :math:`u_1,\dots,u_M`.
        fEvals : list[np.ndarray]
            Precomputed node evaluation :math:`f_1,\dots,f_M`.
        out : np.ndarray
            Output array to store the result.
        """
        assert self.nodes[-1] == 1
        np.copyto(out, uNodes[-1])
        fEvals[0], fEvals[-1] = fEvals[-1], fEvals[0]


    def run(self, uNum=None, tInit=0):
        r"""
        Solve using sequential computation of node solutions for each step,
        using the relation :

        .. math::

            u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}, f_0, f_1, ..., f_{m})
            = u_0.

        and the step update to compute :math:`u(t_0+\Delta_t)` using all
        computed node solutions.


        Parameters
        ----------
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        uNodes = np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
        fEvals = [np.zeros(self.uShape, dtype=self.dtype)
                  for _ in range(self.nNodes+1)]
        self.evalF(uNum[0], self.t0, out=fEvals[0])

        times = self.times + tInit
        tau = self.dt*self.nodes

        for i in range(self.nSteps):

            # initialize first node with starting value for step
            np.copyto(uNodes[0], uNum[i])

            for m in range(self.nNodes):
                self.phiSolve(
                    [uNum[i], *uNodes[:m]], fEvals[:m+1], rhs=uNum[i], out=uNodes[m], t0=times[i])
                self.evalF(u=uNodes[m], t=times[i]+tau[m], out=fEvals[m+1])

            self.stepUpdate(uNum[i], uNodes, fEvals, out=uNum[i+1])

        return uNum


    def runSDC(self, nSweeps, Q=None, weights=None, uNum=None, tInit=0):
        r"""
        Solve the ODE with dense :math:`Q` coefficients using SDC sweeps.

        Considering a **lower-triangular** approximation :math:`Q_\Delta`
        of :math:`Q`, it performs for each time-step :math:`K` SDC sweeps :

        .. math::

            u_{m}^{k+1} - \phi_m^{k+1}
                = u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j)
                - \phi_m^k,

        where
        :math:`\phi_m^k:=\phi(u_0,u_1^k,\dots,u_m^k,f_0,f_1^k,\dots,f_{m-1}^k)`
        and :math:`q_{i,j}` are the coefficients of the :math:`Q` matrix.
        It uses a **copy initialization**, that is :math:`u_{m}^0 = u_0`.

            💡 If we consider that :math:`\phi_m^{k}` is like
            a coarse solver applied on iteration :math:`k` and
            :math:`u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j)` is like
            a fine solver applied to iteration :math:`k`,
            then the SDC correction above furiously resemble to
            a **Parareal iteration** 👻 👻 👻

        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If weights are not used (`weights=False`),
        then it simply uses the last node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        nSweeps : int
            Number of SDC sweeps :math:`K`.
        Q : 2D array-like, optional
            The dense :math:`Q` matrix.
            If not provided, automatically computed using the
            :class:`LagrangeApproximation` class and the solver nodes.
        weights : 1D array-like, optional
            The associated weights :math:`\omega_{m}` for the step update.
            If not provided, automatically computed using the
            :class:`LagrangeApproximation` class and the solver nodes.
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        if Q is None or weights is True:
            approx = LagrangeApproximation(self.nodes)
        if Q is None:
            Q = approx.getIntegrationMatrix([(0, tau) for tau in self.nodes])
        if weights is True:
            weights = approx.getIntegrationMatrix([(0, 1)]).ravel()
        if weights is False:
            weights = None
        nNodes, Q, weights = utils.checkCoeff(Q, weights)
        assert nNodes == self.nNodes, "solver and Q do not have the same number of nodes"
        assert np.allclose(Q.sum(axis=1), self.nodes), "solver and Q do not have the same nodes"
        Q = self.dt*Q
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        uNodes = [np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]
        fEvals = [[np.zeros(self.uShape, dtype=self.dtype)
                   for _ in range(self.nNodes+1)]
                  for _ in range(2)]

        times = self.times + tInit
        tau = self.dt*self.nodes

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            np.copyto(uNodes[0], uNum[i])
            self.evalF(uNum[i], times[i], out=fEvals[0][0])
            np.copyto(fEvals[1][0], fEvals[0][0])   # u_0^{1} = u_0^{0}
            for m in range(self.nNodes):
                np.copyto(fEvals[0][m+1], fEvals[0][0])  # u_m^{k} = u_0^{0}

            uTmp = uNum[i+1]    # use next step as buffer for k correction term

            # loop on sweeps (iterations)
            for _ in range(nSweeps):

                uK0, uK1 = uNodes
                fK0, fK1 = fEvals

                # loop on nodes (stages)
                for m in range(self.nNodes):

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    fK = fK0[1:]  # note : ignore f(u0) term in fK0
                    for j in range(self.nNodes):
                        self.axpy(a=Q[m, j], x=fK[j], y=rhs)

                    # substract k correction term
                    self.evalPhi(
                        [uNum[i], *uK0[:m+1]], fK0[:m+2], out=uTmp, t0=times[i])
                    rhs -= uTmp

                    # solve with k+1 correction
                    self.phiSolve(
                        [uNum[i], *uK1[:m]], fK1[:m+1], out=uK1[m], rhs=rhs, t0=times[i])

                    # evalF on k+1 node solution
                    self.evalF(uK1[m], t=times[i]+tau[m], out=fK1[m+1])

                # invert uK0/fK0 and uK1/fK1 for next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]
                uNodes[0], uNodes[1] = uNodes[1], uNodes[0]

            # step update
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                fK = fK1[1:]  # note : ignore f(u0) term in fK0
                for m in range(self.nNodes):
                    self.axpy(a=weights[m], x=fK[m], y=uNum[i+1])
            else:
                self.stepUpdate(uNum[i], uNodes[0], fEvals[0], out=uNum[i+1])

        return uNum


T = TypeVar("T")

PHI_STEPPERS: dict[str, type[PhiStepper]] = {}
"""Dictionary containing all specialized :class:`PhiTimeStepper` classes"""

def register(cls: type[T]) -> type[T]:
    """Class decorator to register a specialized :class:`PhiTimeStepper` class in `qmint`"""
    checkOverriding(cls, "evalPhi", mroIndex=-3)
    storeClass(cls, PHI_STEPPERS)
    return cls

if __name__ != "__main__":
    # Import all local submodules
    __all__ = [PHI_STEPPERS, register]
    importAll(locals(), __all__, __path__, __name__, __import__)
