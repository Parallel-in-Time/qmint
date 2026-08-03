---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for qmint, with links to the rest of the site..
html_theme.sidebar_secondary.remove: true
---

# QMint Playground

📜 How to use [`qmat`](https://qmat.readthedocs.io) to implement and play with the **many flavors of time-integration** :

- Runge-Kutta methods,
- Spectral Deferred Correction,
- Semi-Lagrangian approach, Multi-step schemes, ...

[![PyPI - Package](https://img.shields.io/pypi/v/qmint?logo=python)](https://pypi.org/project/qmint)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/qmint?logo=pypi&cacheSeconds=86400)](https://pypistats.org/packages/qmint)
[![Last Commit](https://img.shields.io/github/last-commit/parallel-in-time/qmint/main?logo=github)](https://github.com/Parallel-in-Time/qmint)
[![CI pipeline](https://github.com/Parallel-in-Time/qmint/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/Parallel-in-Time/qmint/actions/workflows/ci_pipeline.yml)
[![Codecov](https://codecov.io/gh/Parallel-in-Time/qmint/graph/badge.svg?token=8R927FGCKG)](https://codecov.io/gh/Parallel-in-Time/qmint)

💾 Installation : `pip install qmint`, or see [detailed instructions ...](./installation.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21774191.svg)](https://doi.org/10.5281/zenodo.21774191)

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card}
:link: basics.html
{fas}`book-open;sd-text-primary` Basic usage
^^^
_Using `qmat` to generate Butcher tables and SDC coefficients_
:::

:::{grid-item-card}
:link: advanced.html
{fas}`gear;sd-text-primary` Advanced tutorials
^^^
_Going deeper into time-integration topics with `qmat` and `qmint`_
:::

:::{grid-item-card}
:link: features.html
{fas}`screwdriver-wrench;pst-color-primary` Side-features
^^^
_Additional core features of `qmat` : interpolation, derivation, ..._
:::

:::{grid-item-card}
:link: community.html
{fas}`flask;pst-color-primary` Community playgrounds
^^^
_Shared experiments with `qmat` contributed by the community_
:::

:::{grid-item-card}
:link: api/qmint/index.html
{fas}`scroll;pst-color-primary` API reference
^^^
_Full documentation of the `qmint` companion package_
:::

:::{grid-item-card}
:link: contributing.html
{fas}`bolt;pst-color-primary` Contributing
^^^
_How to contribute to `qmint` playgrounds and codebase_
:::

::::


```{toctree}
:maxdepth: 2
:hidden:

installation
basics
advanced
features
community
API reference <api/qmint/index>
contributing
```