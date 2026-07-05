---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for qmint, with links to the rest of the site..
html_theme.sidebar_secondary.remove: true
---

# QMint Playground

> 📣 Using [`qmat`](https://qmat.readthedocs.io) to implement & play with the **many flavors of time-integration** ...

- Runge-Kutta methods,
- Spectral Deferred Correction,
- Semi-Lagrangian approach, Multi-step schemes, ...

This website centralize tutorials and numerical experiments, from _academical learning_ to _advanced research_.

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
_How can you contribute to the whole `qmat` eco-system_
:::

::::


```{toctree}
:maxdepth: 2
:hidden:

basics
advanced
features
community
API reference <api/qmint/index>
contributing
```