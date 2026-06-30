#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Use `qmat` to implement time-stepper for (non-linear) ODE systems of the form :

.. math::

    \frac{du}{dt} = f(u,t), \quad u(0) = u_0.

All time-steppers are based on a :class:`DiffOp` base class,
implementing :

- the :math:`f(u,t)` evaluations,
- a solver for :math:`u-\alpha f(u,t)=rhs`, considering given :math:`\alpha,t,rhs`.

While the :math:`f(u,t)` evaluations must be implemented,
a default implementation of the solver for :math:`u-\alpha f(u,t)=rhs`
is provided in the base :class:`DiffOp` class.

    🛠️ Various specialized :class:`DiffOp` classes are implemented
    in the :class:`diffops` submodule.

The solvers implemented here discretizes
a time-step :math:`[t_0, t_0+\Delta{t}]` into **time nodes**
:math:`[t_0+\Delta{t}\tau_1, ..., t_0+\Delta{t}\tau_M]`
noted :math:`[t_1,\dots,t_M]`,
also called **stages** for RK methods, at which are defined the
**node solutions** :math:`u_m \simeq u(t_m)`.
And usually, the vector containing the node solutions
:math:`{\bf u} = [u_1,\dots,u_M]^T` satisfy a **all-at-once system** :

.. math::
    {\bf u} - \Delta{t}Q {\bf f} = {\bf u}_0,

where :math:`{\bf f} = [f(u_1, t_1),\dots,f(u_M,t_M)]^T` is the vector
with the evaluations of each node solutions
and :math:`{\bf u}_0` is a vector containing :math:`u_0` in each entry.
The :class:`CoeffStepper` allows to solve any ODE using this coefficient-based
approach, either directly if the :math:`Q` matrix is lower triangular,
or iteratively with SDC-based sweeps if :math:`Q` is a dense matrix.

----

An alternative solver approach relates all the node solutions using a
:math:`\phi` **representation** of a time-integrator,
*i.e* each node solution :math:`u_{m+1}` satisfies
the following relation :

.. math::

    u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}) = u_0,

where :math:`\phi` is solely defined by the chosen time-integrator.
The system above can be solved node-by-node in a sequential approach,
or iteratively with a SDC-based approach.
It is implemented in the abstract :class:`PhiSolver` class,
that needs to be specialized by a child class implementing
the :math:`\phi` function.

    🛠️ Specialized :class:`PhiSolver` classes are implemented in the
    :class:`integrators` submodule.
"""
from qmint.diffops import DiffOp, DIFFOPS
from qmint.steppers.dahlquist import Dahlquist, DahlquistIMEX
from qmint.steppers.coeff import CoeffStepper
from qmint.steppers.phi import PHI_STEPPERS

__all__ = [
    DiffOp, DIFFOPS,
    Dahlquist, DahlquistIMEX,
    CoeffStepper,
    PHI_STEPPERS]