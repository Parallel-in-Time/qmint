# Development Roadmap

📜 _Planned steps for the package development ..._

Detailed description of all specific versions and their associated changes is available on the [Github Releases page](https://github.com/Parallel-in-Time/qmint/releases).

**Status 3 - Alpha** : `v0.0.*`

- integration/documentation of community playgrounds

**Status 4 - Beta** : `v0.1.*`

- external reviews of the basic and advanced tutorials
- addition of a few `DiffOp` classes for experiments with PDEs
- additional tutorials
  - [advanced] different step updates and their impact on the accuracy (from the [`GFM`](https://github.com/Parallel-in-Time/gfm/blob/gaia/notes/GFM_Note-4.ipynb) project)
  - [basics] multilevel SDC
  - [basics] PFASST
  - [features] space-discretization coefficients with `qmat.lagrange` (finite difference, finite volumes, WENO, ...)

**Status 5 - Production/Stable** : `v1.0.*`

- tutorials on the SDC-Butcher theory from J. Fregin (with associated console scripts)
- numerical stability for IMEX schemes

**Status 6 - Mature** : `v1.*.*`

- generic IMEX time-steppers
