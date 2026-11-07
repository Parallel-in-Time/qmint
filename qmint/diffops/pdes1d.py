#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
1D periodic PDE operators in mass-matrix form, for IMEX time-stepping experiments.

The semi-discrete system reads

.. math::

    M \frac{du}{dt} = L u + \mathcal{N}(u),

with, on a uniform periodic mesh of :math:`N` points with spacing :math:`dx`:

* centered finite differences ("FD"): :math:`M=I`, and :math:`L`, :math:`\mathcal{N}`
  are explicit stencil matrices built from the Fourier-symbol tables of
  :mod:`qmint.utils.analysis`;
* Lele compact schemes ("COMPACT"): cyclic tridiagonal systems
  :math:`B_1 f' = R_1 f`, :math:`B_2 f'' = R_2 f`, giving :math:`M = B_1 B_2`,
  :math:`L = \nu B_1 R_2`, :math:`\mathcal{N} = -a B_2 R_1 u` (linear advection)
  or :math:`-B_2 R_1(u^2/2)` (Burgers).  The derivative operators are never formed
  explicitly; only cyclic tridiagonal solves with :math:`B_1`, :math:`B_2` are used.

The classes expose the IMEX splitting used by the generic drivers:
``evalF_IM`` (implicit part :math:`M^{-1} L u`), ``evalF_EX`` (explicit part
:math:`M^{-1} \mathcal{N}(u)`), and ``fSolve_IM(a, rhs, out)`` solving
:math:`(M - a L)u = M\,rhs` (cached sparse LU per :math:`a`).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qmint.diffops import DiffOp, register
from qmint.utils.analysis import ADV_COEFFS, DIFF_COEFFS, ADV_COMPACT, DIFF_COMPACT


def _periodic_stencil(N, stencil):
    """Sparse (N,N) matrix of a periodic stencil [(offset, coeff), ...]."""
    rows, cols, data = [], [], []
    base = np.arange(N)
    for off, c in stencil:
        if c == 0:
            continue
        rows.append(base)
        cols.append((base + off) % N)
        data.append(np.full(N, c))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()


def fd_matrix(deriv, order, N, dx):
    """Periodic centered finite-difference matrix (deriv=1 or 2, order 2/4/6)."""
    table = ADV_COEFFS if deriv == 1 else DIFF_COEFFS
    coeffs, denom = table[order]
    n = len(coeffs)
    # same convention as qmint.utils.analysis.getFourierSymbol:
    # coeffs[0] pairs with exponent +n//2 (matrix column i+e for row i)
    exponents = range(n // 2, -(n // 2) - 1, -1)
    scale = dx if deriv == 1 else dx**2
    stencil = [(e, c / (denom * scale)) for c, e in zip(coeffs, exponents)]
    return _periodic_stencil(N, stencil)


def compact_matrices(deriv, order, N):
    """Cyclic tridiagonal (B, R) of Lele's compact scheme (deriv=1 or 2)."""
    table = ADV_COMPACT if deriv == 1 else DIFF_COMPACT
    alpha, a, b = table[order]
    B = _periodic_stencil(N, [(-1, alpha), (0, 1.0), (1, alpha)])
    if deriv == 1:
        # R1 f = a (f_{i+1}-f_{i-1})/(2dx) + b (f_{i+2}-f_{i-2})/(2dx); dx folded in by caller
        R = _periodic_stencil(N, [(-2, -b / 2), (-1, -a / 2), (1, a / 2), (2, b / 2)])
    else:
        # R2 f = a (f_{i+1}-2f_i+f_{i-1})/dx^2 + b (f_{i+2}-2f_i+f_{i-2})/(4dx^2)
        R = _periodic_stencil(N, [(-2, b / 4), (-1, a), (0, -2 * a - b / 2), (1, a), (2, b / 4)])
    return B, R


class MassMatrixPDE1D(DiffOp):
    r"""
    Base class for 1D periodic mass-matrix PDEs :math:`M u' = L u + \mathcal{N}(u)`.

    Parameters
    ----------
    N : int
        Number of mesh points.
    nu : float
        Diffusion coefficient.
    L : float
        Domain length (periodic interval [0, L)).
    order : int
        Order of the space discretization (2, 4 or 6).
    spaceScheme : str
        "FD" for centered finite differences, "COMPACT" for Lele compact schemes.
    u0 : array-like, optional
        Initial solution. Default is the first sine mode.
    """
    def __init__(self, N=128, nu=1e-2, L=1.0, order=4, spaceScheme="FD", u0=None):
        self.N = N
        self.L = L
        self.dx = L / N
        self.nu = nu
        self.order = order
        self.spaceScheme = spaceScheme.upper()
        x = np.arange(N) * self.dx
        if u0 is None:
            u0 = np.sin(2 * np.pi * x / L)
        super().__init__(u0)
        self.x = x
        self._spluCache = {}
        if self.spaceScheme == "FD":
            self.D1 = fd_matrix(1, order, N, self.dx)
            self.D2 = fd_matrix(2, order, N, self.dx)
            self.M = sp.identity(N, format="csc")
            self.Lmat = nu * self.D2
        elif self.spaceScheme in ("COMPACT", "LELE"):
            self.B1, self.R1 = compact_matrices(1, order, N)
            self.B2, self.R2 = compact_matrices(2, order, N)
            # fold dx scaling into R matrices
            self.R1 = self.R1 / self.dx
            self.R2 = self.R2 / self.dx**2
            self.M = (self.B1 @ self.B2).tocsc()
            self.Lmat = (nu * self.B1 @ self.R2).tocsc()
            # fixed factorizations for the mass-matrix solves
            self._spluB1 = spla.splu(self.B1.tocsc())
            self._spluB2 = spla.splu(self.B2.tocsc())
        else:
            raise ValueError(f"unknown spaceScheme {spaceScheme}")
        # dense forms for the stabilized implicit solves (N is small in 1D)
        self.Mdense = self.M.toarray()
        self.Ldense = self.Lmat.toarray()
        if self.spaceScheme == "FD":
            self.D1eff = self.D1.toarray()
        else:
            # D1eff = M^{-1} (B2 R1) = B1^{-1} R1
            self.D1eff = self._spluB1.solve(self.R1.toarray())

    def stabCref(self, uRef):
        """Frozen A_c^2 coefficient of the Stiller term; overridden in subclasses."""
        return np.ones(self.N)

    def evalF_EX2(self, u, out, uRef=None):
        r"""Second application of the explicit operator with frozen A_c^2:
        out = D1eff (cref * (D1eff u)), i.e. the linearized (theta/2) A_c^2 term."""
        cref = self.stabCref(uRef)
        if self.spaceScheme == "FD":
            out[:] = self.D1 @ (cref * (self.D1 @ u))
        else:
            out[:] = self._spluB1.solve(self.R1 @ (cref * self._spluB1.solve(self.R1 @ u)))

    def fSolve_IM_stab(self, a, theta, rhs, out, uRef=None):
        r"""Solve (M - a L - a (theta/2) M D1eff diag(cref) D1eff) u = M rhs.

        Adds Stiller's Lax-Wendroff term (theta/2) A_c^2 to the implicit
        operator, with A_c^2 frozen (`stabCref`).  `a` and `theta` are both
        dimensional (e.g. a = theta = subinterval size).  Per Fourier mode it
        adds (theta/2)(a/dt_sub...) -- in dimensionless terms (theta/2) zE^2 to
        zI, matching runDahlquistIMEXParStiller (parallel stabilized SDC-PS).
        """
        cref = self.stabCref(uRef)
        A = (self.Mdense - a * self.Ldense
             - a * (theta / 2) * (self.Mdense @ self.D1eff @ (cref[:, None] * self.D1eff)))
        mrhs = self.Mdense @ rhs
        try:
            out[:] = np.linalg.solve(A, mrhs)
        except np.linalg.LinAlgError:
            # non-symmetric stabilized matrix can be singular for some (a, cref)
            out[:] = np.linalg.lstsq(A, mrhs, rcond=None)[0]

    # -- mass-matrix helpers -------------------------------------------------
    def massApply(self, rhs, out):
        """out = M rhs."""
        if self.spaceScheme == "FD":
            np.copyto(out, rhs)
        else:
            out[:] = self.B1 @ (self.B2 @ rhs)

    def massSolve(self, rhs, out):
        """out = M^{-1} rhs (two cyclic tridiagonal solves for compact schemes)."""
        if self.spaceScheme == "FD":
            np.copyto(out, rhs)
        else:
            out[:] = self._spluB2.solve(self._spluB1.solve(rhs))

    # -- IMEX splitting ------------------------------------------------------
    def evalF_IM(self, u, t, out):
        """Implicit part M^{-1} L u (diffusion)."""
        if self.spaceScheme == "FD":
            out[:] = self.Lmat @ u
        else:
            out[:] = self.nu * self._spluB2.solve(self.R2 @ u)

    def evalF_EX(self, u, t, out):
        """Explicit part M^{-1} N(u); overridden in subclasses."""
        raise NotImplementedError

    def evalF(self, u, t, out):
        tmp = np.empty_like(out)
        self.evalF_IM(u, t, out)
        self.evalF_EX(u, t, tmp)
        out += tmp

    def fSolve_IM(self, a, rhs, out):
        r"""Solve (M - a L) u = M rhs for u, storing the result in `out`."""
        mrhs = np.empty_like(rhs)
        self.massApply(rhs, mrhs)
        key = float(a)
        if key not in self._spluCache:
            if self.spaceScheme == "FD":
                A = (sp.identity(self.N, format="csc") - a * self.Lmat).tocsc()
            else:
                # M - a L = B1 (B2 - a nu R2)
                A = (self.B2 - a * self.nu * self.R2).tocsc()
            self._spluCache[key] = spla.splu(A)
        if self.spaceScheme == "FD":
            out[:] = self._spluCache[key].solve(mrhs)
        else:
            out[:] = self._spluCache[key].solve(self.B2 @ rhs)

    def fSolve(self, a, rhs, t, out):
        """Solve u - a f_IM(u) = rhs (implicit part only), for IMEX steppers."""
        self.fSolve_IM(a, rhs, out)


@register
class AdvectionDiffusion1D(MassMatrixPDE1D):
    r"""
    Linear 1D advection-diffusion on a periodic domain:
    :math:`u_t = -a u_x + \nu u_{xx}`, with :math:`\mathcal{N}(u) = -a D_1 u`.

    Parameters
    ----------
    a : float
        Advection speed.
    k : int
        Wave number of the default sine initial condition.
    (see :class:`MassMatrixPDE1D` for the other parameters)
    """
    def __init__(self, a=1.0, k=1, **kwargs):
        self.a = a
        self.k = k
        if "u0" not in kwargs or kwargs["u0"] is None:
            N = kwargs.get("N", 128)
            L = kwargs.get("L", 1.0)
            kwargs["u0"] = np.sin(2 * np.pi * k * np.arange(N) * (L / N) / L)
        super().__init__(**kwargs)

    def evalF_EX(self, u, t, out):
        """Explicit advection -a M^{-1} (B2 R1 u) or -a D1 u."""
        if self.spaceScheme == "FD":
            out[:] = -self.a * (self.D1 @ u)
        else:
            # M^{-1} (B2 R1 u) = B1^{-1} R1 u (B1, B2 circulant commute)
            out[:] = -self.a * self._spluB1.solve(self.R1 @ u)

    def stabCref(self, uRef):
        # linear advection: A_c = a (constant)
        return np.full(self.N, self.a**2)

    def exact(self, T):
        """Exact solution at time T for the sine initial condition."""
        kx = 2 * np.pi * self.k / self.L
        return np.exp(-self.nu * kx**2 * T) * np.sin(kx * (self.x - self.a * T))


@register
class Burgers1D(MassMatrixPDE1D):
    r"""
    1D Burgers equation on a periodic domain:
    :math:`u_t = -\partial_x(u^2/2) + \nu u_{xx}`,
    with :math:`\mathcal{N}(u) = -D_1(u^2/2)` resp. :math:`-B_2 R_1(u^2/2)`.
    """
    def evalF_EX(self, u, t, out):
        """Explicit convective flux -M^{-1} D1(u^2/2)."""
        if self.spaceScheme == "FD":
            out[:] = -(self.D1 @ (0.5 * u**2))
        else:
            out[:] = -self._spluB1.solve(self.R1 @ (0.5 * u**2))

    def stabCref(self, uRef):
        # Burgers: A_c = diag(u), frozen at the step state (Stiller's method)
        if uRef is None:
            uRef = self.u0
        return uRef**2
