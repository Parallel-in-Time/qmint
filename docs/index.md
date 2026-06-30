---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for qmint, with links to the rest of the site..
html_theme.sidebar_secondary.remove: true
---

# QMint Playground

Using [`qmat`](https://github.com/parallel-in-Time/qmat) to implement & play with the many flavors of time-integration (Spectral Deferred Correction, Runge-Kutta, ...)


::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card}
:link: basics.html
{fas}`spinner;sd-text-primary` Basic usage
^^^
_From Butcher Tables to Spectral Deferred Correction methods ... basic usage examples of `qmat`_
:::

:::{grid-item-card}
:link: advanced.html
{fas}`spinner;sd-text-primary` Advanced tutorials
^^^
_Going deeper into advanced time-integration topics with `qmat` and `qmint`_
:::

:::{grid-item-card}
:link: features.html
{fas}`bolt;pst-color-primary` Side-features
^^^
_Because `qmat` can do way more than just time-integration thanks to its core features_
:::

:::{grid-item-card}
:link: community.html
{fas}`bolt;pst-color-primary` Community playgrounds
^^^
_Shared experiments with `qmat` contributed by the community_
:::

:::{grid-item-card}
:link: api/qmint/index.html
{fas}`bolt;pst-color-primary` API reference
^^^
Full documentation of `qmint`
:::

:::{grid-item-card}
:link: contributing.html
{fas}`bolt;pst-color-primary` Contributing
^^^
How can you contribute to the whole `qmat` eco-system
:::

::::


## Doc Contents

```{toctree}
:maxdepth: 2

basics
advanced
features
community
API reference <api/qmint/index>
contributing
```