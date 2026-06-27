#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TODO : ...
"""
import numpy as np
import scipy.optimize as sco
import warnings

from typing import TypeVar
from qmat.utils import checkOverriding, storeClass, importAll


class DiffOp():
    r"""
    Base class for a differential operator :math:`f(u, t)` used in a generic ODE.

    It defines the evaluation of :math:`f(u, t)` at given :math:`u` and
    :math:`t` with a `evalF(u, t, out)` method, that put the result
    of the evaluation in the `out` array.

    Additionally, this class defines a default `fSolve` method that solves :

    .. math::

        u - \alpha f(u,t) = rhs

    for given :math:`\alpha`, :math:`t` and :math:`rhs`.
    This default method can be overridden by a more efficient specific
    method for a specific differential operator.

    Note
    ----
    Solutions are stored in N-dimensional :class:`numpy.ndarray`.

    Parameters
    ----------
    u0 : array-like
        The initial solution associated to the differential operator, to which
        is extracted the generic shape and datatype of :math:`u(t)` solutions.
    """
    def __init__(self, u0):
        for name in ["u0", "innerSolver"]:
            assert not hasattr(self, name), \
                f"{name} attribute is reserved for the base DiffOp class"
        self.u0 = np.asarray(u0)
        """Initial solution for the differential operator."""
        if self.u0.size < 1e3:
            self.innerSolver = sco.fsolve
            """Inner solver used in the default `fSolve` method."""
        else:
            self.innerSolver = sco.newton_krylov

    @property
    def uShape(self):
        """Shape of a :math:`u` solution, stored as numpy array."""
        return self.u0.shape

    @property
    def dtype(self):
        """Datatype of a :math:`u` solution, stored as numpy array."""
        return self.u0.dtype


    def evalF(self, u:np.ndarray, t:float, out:np.ndarray):
        """
        Evaluate :math:`f(u,t)` and store the result into `out`.

        Parameters
        ----------
        u : np.ndarray
            Input solution for the evaluation.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Output array in which is stored the evaluation.
        """
        raise NotImplementedError("evalF must be provided")


    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        r"""
        Solve :math:`u-\alpha f(u,t)=rhs` for given :math:`u,t,rhs`,
        using `out` as initial guess and storing the final result into it.

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
        def func(u:np.ndarray):
            """compute res = u - a*f(u,t) - rhs"""
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalF(u, t, out=res)
            res *= -a
            res += u
            res -= rhs
            return res.ravel()

        sol = self.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    @classmethod
    def test(cls, t0=0, dt=1e-1, eps=1e-3, instance=None):
        """
        Class method to test the `DiffOp` implementation.

        Parameters
        ----------
        t0 : float, optional
            Evaluation time to test the instance. The default is 0.
        dt : float, optional
            Time-step to test the `fSolve` method. The default is 1e-1.
        eps : float, optional
            Perturbation added in the expected solution to test the
            `fSolve` method. The default is 1e-3.
        instance :`DiffOp`, optional
            Instance to be tested. If not provided (`None`),
            an instance is created using the default constructor.
        """
        if instance is None:
            try:
                instance = cls()
            except:
                raise TypeError(f"{cls} cannot be instantiated with default parameters")

        u0 = instance.u0
        try:
            uEval = np.zeros_like(u0)
            instance.evalF(u=u0, t=t0, out=uEval)
        except:
            raise ValueError("evalF cannot be properly evaluated into an array like u0")

        try:
            uEval *= -dt
            uEval += u0
            uSolve = np.copy(u0)
            uSolve += eps*np.linalg.norm(uSolve, np.inf)
            instance.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)
        except:
            raise ValueError("fSolve cannot be properly evaluated into an array like u0")
        np.testing.assert_allclose(
            uSolve, u0, err_msg="fSolve does not satisfy the fixed-point problem with u0",
            atol=1e-15)

        # check for nan acceptation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uSolve[:] = np.nan
            instance.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)

T = TypeVar("T")

DIFFOPS: dict[str, type[DiffOp]] = {}
"""Dictionary containing all specialized :class:`DiffOp` classes"""

def register(cls: type[T]) -> type[T]:
    """Class decorator to register a specialized :class:`DiffOp` class in `qmint`"""
    checkOverriding(cls, "evalF")
    storeClass(cls, DIFFOPS)
    return cls

if __name__ != "__main__":
    # Import all local submodules
    __all__ = [DIFFOPS, register]
    importAll(locals(), __all__, __path__, __name__, __import__)
