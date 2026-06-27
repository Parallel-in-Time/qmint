import numpy as np


def checkCoeff(Q, weights):
    """
    Check :math:`Q` coefficients and associated weights for a time-integration scheme.

    Parameters
    ----------
    Q : 2D array-like
        The :math:`Q` coefficients.
    weights : 1D array-like
        Quadrature weights associated to the nodes.

    Returns
    -------
    nNodes : int
        Number of nodes (stages).
    Q : np.2darray
        The :math:`Q` coefficients.
    weights : np.1darray
        Quadrature weights associated to the nodes.
    """
    Q = np.asarray(Q)
    nNodes = Q.shape[0]
    assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, \
            f"weights must be a 1D vector, not {weights}"
        assert weights.size == nNodes, \
            "weights size is not the same as the node size"
        assert np.allclose(weights.sum(), 1), \
            "weights sum must be equal to 1"
    else:
        assert np.allclose(Q.sum(axis=1)[-1], 1), \
            "last node must be 1 if weights are not given"

    return nNodes, Q, weights


def checkCoeffSDC(Q, weights, QDelta, nSweeps):
    r"""
    Check SDC coefficients

    Parameters
    ----------
    Q : 2D array-like
        The :math:`Q` coefficients.
    weights : 1D array-like
        Quadrature weights associated to the nodes.
    QDelta : 2D or 3D array-like
        The :math:`Q_\Delta` coefficients (3D if changes with sweeps).
    nSweeps : int
        Number of sweeps.

    Returns
    -------
    nNodes : int
        Number of nodes.
    Q : np.2darray
        The :math:`Q` coefficients.
    weights : np.1darray
        Quadrature weights associated to the nodes.
    QDelta : np.2darray
        The :math:`Q_\Delta` coefficients for each sweep.
    nSweeps : int
        The number of sweeps.
    """
    Q = np.asarray(Q)
    nodes = Q.sum(axis=1)
    nNodes = nodes.size
    assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, "weights must be a 1D vector"
        assert weights.size == nNodes, \
            "weights size is not the same as the node size"
    else:
        assert np.allclose(nodes[-1], 1), \
            "last node must be 1 if weights are not given"

    QDelta = np.asarray(QDelta)
    if QDelta.ndim == 3:
        assert QDelta.shape == (nSweeps, nNodes, nNodes), \
            "inconsistent shape for QDelta"
    else:
        assert QDelta.shape == (nNodes, nNodes), \
            "inconsistent shape for QDelta"
        QDelta = np.repeat(QDelta[None, ...], nSweeps, axis=0)

    return nNodes, Q, weights, QDelta, nSweeps


def checkCoeff_IMEX(QI, wI, QE, wE):
        r"""
        Check IMEX :math:`Q` coefficients and assert their consistency.

        Parameters
        ----------
        QI : 2D array-like
            :math:`Q` coefficients used for :math:`\lambda_I`.
        wI : 1D array-like or None
            Weights used for the step update on :math:`\lambda_I`.
            If None, then step update is not done.
        QE : 2D array-like
            :math:`Q` coefficients used for :math:`\lambda_E`.
        wE : 1D array-like or None
            Weights used for the step update on :math:`\lambda_E`.
            If None, then step update is not done.

        Returns
        -------
        nNodes : int
            Number of nodes.
        QI : np.2darray
            :math:`Q` coefficients used for :math:`\lambda_I`.
        wI : np.1darray or None
            Weights used for the step update on :math:`\lambda_I`.
        QE : np.2darray
            :math:`Q` coefficients used for :math:`\lambda_E`.
        wE : np.1darray or None
            Weights used for the step update on :math:`\lambda_E`.
        useWeights : boll
            Wether or not the step update (using weights) is done.
        """
        QI, QE = np.asarray(QI), np.asarray(QE)
        assert np.allclose(QI.sum(axis=1), QE.sum(axis=1)), \
            "QI and QE do not correspond to the same nodes"

        nNodes = QI.shape[0]
        assert QI.shape == (nNodes, nNodes), "QI is not a square matrix"
        assert QI.shape == QE.shape, "QI and QE do not have the same shape"

        useWeights = True
        if wI is None or wE is None:
            assert wE is None and wI is None, \
                "it's either weights for everyone or no weight"
            useWeights = False

        return nNodes, QI, wI, QE, wE, useWeights


def checkCoeffSDC_IMEX(Q, weights, QDeltaI, QDeltaE, nSweeps):
    r"""
    Check coefficients given for a IMEX SDC sweeps

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

    Returns
    -------
    nNodes : int
        Number of nodes.
    Q : np.2darray
        The :math:`Q` coefficients.
    weights : np.1darray
        Quadrature weights associated to the nodes.
    QDeltaI : np.3darray
        The :math:`Q_\Delta^I` coefficients used for the :math:`\lambda_I`
        term for each sweeps.
    QDeltaE : np.3darray
        The :math:`Q_\Delta^E` coefficients used for the :math:`\lambda_E`
        term for each sweeps.
    nSweeps : int
        Number of SDC sweeps.
    """
    Q = np.asarray(Q)
    nodes = Q.sum(axis=1)
    nNodes = nodes.size
    assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, "weights must be a 1D vector"
        assert weights.size == nNodes, \
            "weights size is not the same as the node size"

    QDeltaI = np.asarray(QDeltaI)
    QDeltaE = np.asarray(QDeltaE)
    if QDeltaI.ndim == 3:
        assert QDeltaI.shape == (nSweeps, nNodes, nNodes), \
            "inconsistent shape for QDeltaI"
    else:
        assert QDeltaI.shape == (nNodes, nNodes), \
            "inconsistent shape for QDeltaE"
        QDeltaI = np.repeat(QDeltaI[None, ...], nSweeps, axis=0)
    if QDeltaE.ndim == 3:
        assert QDeltaE.shape == (nSweeps, nNodes, nNodes), \
            "inconsistent shape for QDeltaE"
    else:
        assert QDeltaE.shape == (nNodes, nNodes), \
            "inconsistent shape for QDeltaE"
        QDeltaE = np.repeat(QDeltaE[None, ...], nSweeps, axis=0)

    return nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps