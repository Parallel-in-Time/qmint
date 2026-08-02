# Contributing

📜 _Package is currently developed with an open-source philosophy, so any relevant contribution is welcome_

## General rules

Ideally, your code implementation should be :

- _**simple** enough so your grandma can understand it_
- _**beautiful** enough to make your cousin in art school want to read it_
- _**efficient** enough such that you spend more time analyzing your plots than coding and running experiments_

Of course, that's an ideal goal ... but nothing prevents to aim at the sky when reaching the top of the mountain 🚡

Recommended approach is to **fork this repository**, create a new branch in your fork named with the reason of your PR
(don't use `main` !), and open a **pull request** when ready.
This will automatically trigger the CI pipeline that :

1. check linting with `flake8`
2. run all the tests defined in the [`tests` folder](https://github.com/Parallel-in-Time/qmint/tree/main/tests), and upload a coverage report to [`codecov`](https://app.codecov.io/gh/Parallel-in-Time/qmint)
3. test all the notebooks located in [`docs/basics`](https://github.com/Parallel-in-Time/qmint/tree/main/docs/basics), [`docs/advanced`](https://github.com/Parallel-in-Time/qmint/tree/main/docs/advanced) and [`docs/features`](https://github.com/Parallel-in-Time/qmint/tree/main/docs/features)

Current coverage is at 100%, so no untested line will be accepted 😇.

> 📣 Know that no code styling formatter (like `black`, or else ...) will ever be imposed in CI, as long as I'm still breathing !

Chosen merge strategy is to squash commits $\Rightarrow$ you don't have to care about the number of commit included in your PR, so don't be scare of making mistakes before your PR is accepted 😉

> 🔔 Once your PR is accepted, please delete this branch from your fork and synchronize your `main` branch. When creating a new development branch later, ensure that you start from an up-to-date `main` branch of your fork.

In case you are interested in contributing but don't have any idea on what, checkout out current [development roadmap 🎯](./contrib/roadmap.md) and [project proposals 🎓](https://github.com/Parallel-in-Time/qmint/discussions/categories/project-proposals)

## Base recipes

_Some memos on how to develop this package ..._

- [Add a playground](./contrib/addPlayground.md)
- [Add a differential operator](./contrib/addDiffOp.md)
- [Add a $\phi$-based time-integrator](./contrib/addPhiIntegrator.md)
- [Testing your changes](./contrib/testing.md)
- [Update this documentation](./contrib/updateDoc.md)
- [Version update pipeline](./contrib/versionUpdate.md)

```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:

    contrib/addPlayground
    contrib/addDiffOp
    contrib/addPhiIntegrator
    contrib/testing
    contrib/updateDoc
    contrib/versionUpdate
    contrib/roadmap
```

## Miscellaneous

```{eval-rst}
.. toctree::
    :maxdepth: 1

    CODE_OF_CONDUCT
    SECURITY
```
