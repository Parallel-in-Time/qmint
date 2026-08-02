# Add a $\phi$-based time-stepper

📜 _Additional time schemes can be added using the [$\phi$ formulation](../advanced/4_phiIntegrator.ipynb)_
_to test other variants of $Q_\Delta$-coefficients free Spectral Deferred Correction._
_For that, you can implement a new {py:mod}`PhiStepper <qmint.steppers.phi.PhiStepper>` class in the {py:mod}`qmint.steppers.phi` subpackage_.

Add your class in a module of the {py:mod}`qmint.steppers.phi` sub-package using the following template :

```python
class Phidlidoo(PhiStepper):
    r"""
    Base description, in particular its definition :

    .. math::

        \phi(u_0, u_1, ..., u_{m}, u_{m+1}) =
            ...

    And eventual parameters description ...
    """

    def evalPhi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) in [m, m+1]

        # TODO : integrators implementation
```

The first assertions are not mandatory, but ensure that the `evalPhi` is properly evaluated.

> 📣 New `PhiStepper` classes are not automatically tested, so you'll have to write
> dedicated tests for your new class in `tests.test_stepper.test_phi.py`.
> Checkout those already implemented for `ForwardEuler` and `BackwardEuler`.

As for the {py:class}`DiffOp <qmint.diffops.DiffOp>` class,
the {py:class}`PhiStepper <qmint.steppers.phi.PhiStepper>` implements a generic default
`phiSolve` method, that you can override by a more efficient specialized approach.

> 💡 Note that the model above inherits the `__init__` constructor of the `PhiStepper` class,
> so it can take any `DiffOp` class as parameter.
> If your time-integrator is specialized for some kind of differential operators
> (_e.g_ a semi-Lagrangian scheme for an advective problem),
> then you probably need to override the `__init__` method in your class too.
