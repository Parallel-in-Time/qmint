#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Time-steppers for the Dahlquist equation based on :math:`Q` coefficients,
also implementing SDC sweeps with given :math:`Q_\Delta` coefficients.
"""
import numpy as np

import qmint.utils as utils


class Dahlquist():
    r"""
    Solver for the classical Dahlquist equation

    .. math::

        \frac{du}{dt} = \lambda u, \quad u(0)=u_0, \quad t \in [0,T].

    It can be used to solve the equation with multiple :math:`\lambda`
    values (multiple trajectories) using efficient vectorized
    computation.
    Furthermore, it has no restriction on the used
    :math:`Q` and :math:`Q_\Delta` matrices (can be dense),
    which is not the case for the generic
    :class:`CoeffSolver` used with
    :class:`qmat.solvers.generic.diffops.Dahlquist`.

    Parameters
    ----------
    lam : scalar or array
        Value(s) used for :math:`\lambda`.
    u0 : scalar or array, optional
        Initial value :math:`\lambda`, must be compatible with `lam`.
        The default is 1.
    tEnd : float, optional
        Final simulation time :math:`T`. The default is 1.
    nSteps : float, optional
        Number of time-step to solve. The default is 1.
    """
    def __init__(self, lam, u0=1, tEnd=1, nSteps=1):
        self.u0 = u0
        """initial solution value"""

        self.tEnd = tEnd
        """final simulation time"""

        self.nSteps = nSteps
        """number of time-steps"""

        self.dt = tEnd/nSteps
        """time-step size"""

        self.lam = np.asarray(lam)
        r"""array storing the :math:`\lambda` values"""
        try:
            lamU = self.lam*u0
        except:
            raise ValueError("error when computing lam*u0")
        self.uShape = tuple(lamU.shape)
        """shape of the solution at a given time"""
        self.dtype = lamU.dtype
        """solution datatype"""


    def run(self, Q, weights):
        r"""
        Run the time-stepper for all :math:`\lambda` using a direct solve of the :math:`Q` matrix.
        For each time-step it solves :

        .. math::

            (I - \Delta{t}\lambda Q){\bf u} = {\bf u}_0,

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry.
        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u},

        or simply use the last **node solution** :math:`{\bf u}[-1]` if
        no weights are given (`weights=None`).

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or None
            Quadrature weights associated to the nodes.
            If None, do not use them for the step update
            (requires last node equal to 1)

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights = utils.checkCoeff(Q, weights)

        # Collocation problem matrix
        A = np.eye(nNodes) - self.lam[..., None, None]*self.dt*Q

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):
            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]
            if weights is not None:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lam[..., None]*uNodes, weights)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum


    def runSDC(self, Q, weights, QDelta, nSweeps):
        r"""
        Run the time-stepper for all :math:`\lambda` using SDC sweeps.
        For each time-step and sweep :math:`k`, it solves :

        .. math::

            (I - \Delta{t}\lambda Q_\Delta){\bf u}^{k+1}
            = {\bf u}_0 + \Delta{t}\lambda(Q - Q_\Delta){\bf u}^{k},

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry and :math:`{\bf u}^0 = {\bf u}_0`
        (copy initialization).

        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u}^{K},

        where :math:`K` is the total number of sweeps.
        If no weights are given (`weights=None`), it simply uses the last
        **node solution** :math:`{\bf u}[-1]`.

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or None
            Quadrature weights associated to the nodes.
            If None, do not use them for the step update
            (requires last node equal to 1)
        QDelta : 2D or 3D array-like
            The :math:`Q_\Delta` coefficients (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights, QDelta, nSweeps = utils.checkCoeffSDC(
            Q, weights, QDelta, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - self.lam[..., None, None, None]*self.dt*QDelta

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lam[..., None, None]*self.dt*(Q - QDelta[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lam[..., None]*uNodes, weights)

        return uNum


class DahlquistIMEX():
    r"""
    Time-stepper for the IMEX Dahlquist equation

    .. math::

        \frac{du}{dt} = (\lambda_I + \lambda_E) u,
        \quad u(0)=u_0, \quad t \in [0,T].

    It can be used to solve the equation with multiple :math:`\lambda_I`
    and / or :math:`\lambda_E` values (multiple trajectories).

    Parameters
    ----------
    lamI : TYPE
        Value(s) used for :math:`\lambda_I`..
    lamE : scalar or array
        Value(s) used for :math:`\lambda_E`.
    u0 : scalar or array, optional
        Initial value :math:`\lambda`, must be compatible with `lam`.
        The default is 1.
    tEnd : float, optional
        Final simulation time :math:`T`. The default is 1.
    nSteps : float, optional
        Number of time-step to solve. The default is 1.
    """
    def __init__(self, lamI, lamE, u0=1, tEnd=1, nSteps=1):
        self.u0 = u0
        """initial solution value"""

        self.tEnd = tEnd
        """final simulation time"""

        self.nSteps = nSteps
        """number of time-steps"""

        self.dt = tEnd/nSteps
        """time-step size"""

        self.lamI = np.asarray(lamI)
        r"""array storing the :math:`\lambda_I` values"""
        self.lamE = np.asarray(lamE)
        r"""array storing the :math:`\lambda_E` values"""
        try:
            lamU = (self.lamI + self.lamE)*u0
        except:
            raise ValueError("error when computing (lamI + lamE)*u0")
        self.uShape = tuple(lamU.shape)
        """shape of the solution at one given time"""
        self.dtype = lamU.dtype
        """datatype of the solution array"""


    def run(self, QI, wI, QE, wE):
        r"""
        Run the time-stepper for all :math:`\lambda_I` and :math:`\lambda_E` using direct solve of the :math:`Q^I` and :math:`Q^E` matrices.
        For each time-step it solves :

        .. math::

            (I - \lambda_I Q^I - \lambda_E Q^E){\bf u} = {\bf u}_0

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry.
        The next step solution is computed using the IMEX **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda_I{\bf w}_I^T{\bf u}
            + \Delta{t}\lambda_E{\bf w}_E^T{\bf u},

        or simply use the last **node solution** :math:`{\bf u}[-1]` if
        no weights are given (`wI=wE=None`).

        Parameters
        ----------
        QI : 2D array-like
            :math:`Q^I` coefficients used for :math:`\lambda_I`.
        wI : 1D array-like or None
            Weights used for the step update on :math:`\lambda_I`.
            If None, then step update is not done.
        QE : 2D array-like
            :math:`Q^E` coefficients used for :math:`\lambda_E`.
        wE : 1D array-like or None
            Weights used for the step update on :math:`\lambda_E`.
            If None, then step update is not done.

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, QI, wI, QE, wE, useWeights = utils.checkCoeff_IMEX(QI, wI, QE, wE)

        # Collocation problem matrix
        A = np.eye(nNodes) \
            - self.lamI[..., None, None]*self.dt*QI \
            - self.lamE[..., None, None]*self.dt*QE

        # Solution vector for each time-step
        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        # Time-stepping loop
        for i in range(self.nSteps):

            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]

            if useWeights:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lamI[..., None]*uNodes, wI)
                uNum[i+1] += self.dt*np.dot(self.lamE[..., None]*uNodes, wE)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum


    def runSDC(self, Q, weights, QDeltaI, QDeltaE, nSweeps, theta=0, epsilon=0):
        r"""
        Run the time-stepper for all :math:`\lambda_I` and :math:`\lambda_E` using SDC sweeps.
        For each time-step and sweep :math:`k` it solves :

        .. math::

            (I - \Delta{t}\lambda_I Q_\Delta^I - \Delta{t}\lambda_E Q_\Delta^I){\bf u}^{k+1}
            = {\bf u}_0 + \Delta{t}\left[
                \lambda Q - \lambda_I Q_\Delta^I - \lambda_E Q_\Delta^E\right]
            {\bf u}^{k},

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry and :math:`{\bf u}^0 = {\bf u}_0`
        (copy initialization).
        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u}^{K},

        where :math:`K` is the total number of sweeps.
        If no weights are given (`weights=None`), it simply uses the last
        **node solution** :math:`{\bf u}[-1]`.

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or none
            Quadrature weights associated to the nodes. If None, last node is
            used for the step update.
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^I` coefficients used for the :math:`\lambda_I`
            term (3D if changes with sweeps).
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^E` coefficients used for the :math:`\lambda_E`
            term (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.
        theta : float
            Coefficient for the term :math:`\frac{\theta}{2}\lambda_E^2` that is
            added to :math:`\lambda_I` (default is 0)
        epsilon : float
            Coefficient for the multiplicative factor :math:`(1+\epsilon)`
            for :math:`\lambda_I` (default is 0)

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps = utils.checkCoeffSDC_IMEX(
            Q, weights, QDeltaI, QDeltaE, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - (self.lamI[..., None, None, None]*(1+epsilon)
               + theta/2*self.lamE[..., None, None, None]**2
               )*self.dt*QDeltaI \
            - self.lamE[..., None, None, None]*self.dt*QDeltaE

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lamI[..., None, None]*self.dt*(Q - QDeltaI[k]) @ uNodes \
                    + self.lamE[..., None, None]*self.dt*(Q - QDeltaE[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(
                    (self.lamI[..., None] + self.lamE[..., None])*uNodes, weights)

        return uNum
