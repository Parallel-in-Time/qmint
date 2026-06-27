import numpy as np
from scipy.linalg import blas

import qmint.utils as utils
from qmint.diffops import DiffOp


class CoeffStepper():
    r"""
    Time-stepper for generic (non-linear) ODE system using coefficient-based schemes.

    It considers the following ODE :

    .. math::
        \frac{du}{dt} = f(u,t), \quad u(0)=u_0,

    and solve it on discrete time-steps :math:`0=t_0 < t_1 < \dots < t_N = T_{end}`
    using equidistant time-step size :math:`\Delta{t}`.
    For each time-steps, it uses internal computation stages defined by coefficients,
    like :math:`Q` matrix, Butcher table, ...

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    tEnd : float, optional
        Final simulation time :math:`T_{end}`. The default is 1.
    nSteps : int, optional
        Number :math:`N` of time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time :math:`t_0`. The default is 0.
    """
    def __init__(self, diffOp:DiffOp, tEnd=1, nSteps=1, t0=0):
        assert isinstance(diffOp, DiffOp)
        self.diffOp = diffOp
        """Differential Operator implementing :math:`f(u,t)`."""
        self.axpy = blas.get_blas_funcs('axpy', dtype=self.dtype)
        r"""BLAS-I function executing :math:`y=\alpha x + y` for any solution vectors :math:`x,y`."""

        self.t0 = t0
        """Initial simulation time."""
        self.tEnd = tEnd
        """Final simulation time."""
        self.nSteps = nSteps
        """Number of simulation time-steps"""
        self.dt = (tEnd-t0)/nSteps
        """Time-step size for the simulation"""

    @property
    def u0(self):
        """Initial solution for the problem"""
        return self.diffOp.u0

    @property
    def uShape(self):
        """Shape of the solution at a given time."""
        return self.diffOp.uShape

    @property
    def dtype(self):
        """Datatype of the solution at a given time."""
        return self.diffOp.dtype

    @property
    def times(self):
        """Time values for each time-step"""
        return np.linspace(self.t0, self.tEnd, self.nSteps+1)

    def evalF(self, u:np.ndarray, t:float, out:np.ndarray):
        """
        Wrapper for the `DiffOp` function evaluating :math:`f(u,t)`.

        Parameters
        ----------
        u : np.ndarray
            Input solution for the evaluation.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Output array in which is stored the evaluation.
        """
        self.diffOp.evalF(u, t, out)


    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        r"""
        Wrapper for the `DiffOp` function solving :math:`u-\alpha f(u,t) = rhs`.

        Parameters
        ----------
        a : float
            The :math:`\alpha` coefficient.
        rhs : np.ndarray
            The right hand side.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Input-output array used as initial guess,
            in which is stored the solution.
        """
        self.diffOp.fSolve(a, rhs, t, out)


    @staticmethod
    def lowerTri(Q:np.ndarray, strict=False):
        """
        Check if a 2D matrix is lower triangular.

        Parameters
        ----------
        Q : np.ndarray
            Matrix to check.
        strict : bool, optional
            Check for strictly lower triangular matrix. The default is False.

        Returns
        -------
        bool
            Is the matrix (strictly) lower triangular or not.
        """
        return np.allclose(np.triu(Q, k=0 if strict else 1), np.zeros(Q.shape))


    def run(self, Q, weights, uNum=None, tInit=0):
        r"""
        Solve the ODE considering **lower-triangular** :math:`Q` coefficients.

        This is equivalent to the classical implementation of a generic
        Runge-Kutta method using its Butcher table.
        For each time-step, it defines a node solution (or stage)
        :math:`u_{m}` that is solved using previously computed
        node solution :

        .. math::

            u_{m} - \Delta{t}q_{m,m}f(u_m,t_m)
            = u_0 + \Delta{t}\sum_{j=1}^{m-1}q_{m,j}f(u_j, t_j),

        where :math:`t_m = t_0 + \tau_m` and :math:`q_{i,j}`
        are the coefficients :math:`Q`.
        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If no weights are provided, then it simply uses the last
        node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        Q : np.2darray-like
            The **lower-triangular** :math:`Q`-coefficients matrix.
        weights : np.1darray-like
            The associated :math:\omega_{m}` weights. If not provided,
            use the last node solution for the update
            (requires :math:`\tau_{M} = 1`).
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
        nNodes, Q, weights = utils.checkCoeff(Q, weights)

        assert self.lowerTri(Q), "lower triangular matrix Q expected for non-linear solver"
        Q = self.dt*Q
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0
        assert np.shape(uNum) == (self.nSteps+1, *self.uShape), \
            "user-provided uNum do not have the correct shape"

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = np.zeros((nNodes, *self.uShape), dtype=self.dtype)

        times = self.times + tInit
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):
            uNode = uNum[i+1]
            np.copyto(uNode, uNum[i])

            # loop on nodes (stages)
            for m in range(nNodes):
                tNode = times[i]+tau[m]

                # build RHS
                np.copyto(rhs, uNum[i])
                for j in range(m):
                    self.axpy(a=Q[m, j], x=fEvals[j], y=rhs)

                # solve node (if non-zero diagonal coefficient)
                if Q[m, m] != 0:
                    self.fSolve(a=Q[m, m], rhs=rhs, t=tNode, out=uNode)
                else:
                    np.copyto(uNode, rhs)

                # evalF on current store stage
                self.evalF(u=uNode, t=tNode, out=fEvals[m])

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fEvals[m], y=uNum[i+1])

        return uNum


    def runSDC(self, nSweeps, Q, weights, QDelta, uNum=None, tInit=0):
        r"""
        Solve the ODE with dense :math:`Q` coefficients using SDC sweeps.

        Considering a **lower-triangular** approximation :math:`Q_\Delta`
        of :math:`Q`, it performes for each time-step :math:`K` SDC sweeps :

        .. math::

            \begin{aligned}
            u_{m}^{k+1} - \Delta{t}q^\Delta_{m,m}f(u_m^{k+1},t_m)
                =&~ u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j) \\
            &+ \Delta{t}\sum_{j=1}^{m-1}q^\Delta_{m,j}f(u_j^{k+1},t_j)
            - \Delta{t}\sum_{j=1}^{m}q^\Delta_{m,j}f(u_j^{k},t_j),
            \end{aligned}

        where :math:`q^\Delta_{i,j}` and :math:`q_{i,j}` are the coefficients
        of :math:`Q_\Delta` and :math:`Q`, respectively.
        It uses a **copy initialization**, that is :math:`u_{m}^0 = u_0`.

        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If no weights are provided, then it simply uses the last
        node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        nSweeps : int
            Number of SDC sweeps :math:`K`.
        Q : 2D array-like
            The dense :math:`Q` matrix.
        weights : 1D array-like
            The associated weights :math:`\omega_{m}` for the step update.
        QDelta : 2D array-like
            The lower-triangular :math:`Q_\Delta` matrix.
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
        nNodes, Q, weights, QDelta, nSweeps = utils.checkCoeffSDC(Q, weights, QDelta, nSweeps)
        for qDelta in QDelta:
            assert self.lowerTri(qDelta), \
                "lower triangular matrices QDelta expected for non-linear SDC solver"

        Q, QDelta = self.dt*Q, self.dt*QDelta
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = [np.zeros((nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]

        times = self.times + tInit
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            self.evalF(u=uNum[i], t=times[i], out=fEvals[0][0])
            np.copyto(fEvals[0][1:], fEvals[0][0])

            uNode = uNum[i+1]

            # loop on sweeps (iterations)
            for k in range(nSweeps):
                np.copyto(uNode, uNum[i])

                fK0 = fEvals[0]
                fK1 = fEvals[1]
                qDelta = QDelta[k]

                # loop on nodes (stages)
                for m in range(nNodes):
                    tNode = times[i] + tau[m]

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    for j in range(nNodes):
                        self.axpy(a=Q[m, j], x=fK0[j], y=rhs)

                    # add correction terms (from previous nodes)
                    for j in range(m):
                        self.axpy(a= qDelta[m, j], x=fK1[j], y=rhs)
                        self.axpy(a=-qDelta[m, j], x=fK0[j], y=rhs)

                    # diagonal term (current node)
                    if qDelta[m, m] != 0:
                        self.axpy(a=-qDelta[m, m], x=fK0[m], y=rhs)
                        self.fSolve(a=qDelta[m, m], rhs=rhs, t=tNode, out=uNode)
                    else:
                        np.copyto(uNode, rhs)

                    # evalF on current node
                    self.evalF(u=uNode, t=tNode, out=fK1[m])

                # invert fK0 and fK1 for the next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fK1[m], y=uNum[i+1])

        return uNum
