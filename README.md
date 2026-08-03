# QMint Playground

[![Read the Docs](https://img.shields.io/readthedocs/qmint?logo=readthedocs)](https://qmint.readthedocs.io/)
[![Repo status](https://www.repostatus.org/badges/latest/active.svg)](https://github.com/Parallel-in-Time/qmint)
[![CI pipeline](https://github.com/Parallel-in-Time/qmint/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/Parallel-in-Time/qmint/actions/workflows/ci_pipeline.yml)
[![codecov](https://codecov.io/gh/Parallel-in-Time/qmint/graph/badge.svg?token=8R927FGCKG)](https://codecov.io/gh/Parallel-in-Time/qmint)
[![PyPI - Package](https://img.shields.io/pypi/v/qmint?logo=python)](https://pypi.org/project/qmint)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/qmint?logo=pypi&cacheSeconds=86400)](https://pypistats.org/packages/qmint)

`qmint` is a companion package of [`qmat`](https://github.com/parallel-in-Time/qmat) to play with the many flavors of time-integration methods, like
Spectral Deferred Corrections (SDC), Runge-Kutta methods (RKM), or `qmat` features in general.

📜 This project has two main goals

- provide an [documentation website](https://qmint.readthedocs.io) on the **theory** and **implementation** of SDC-related time-integration methods
- implement a [companion package](./qmint) containing small classes and functions for quick **experiment** and **analysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21774191.svg)](https://doi.org/10.5281/zenodo.21774191)

## Installation

```bash
pip install qmint
```

🔍 See more [detailed instructions for conda environment, development, ...](https://qmint.readthedocs.io/en/latest/installation.html)

## Links

- Documentation : https://qmint.readthedocs.io/
- Issues Tracker : https://github.com/Parallel-in-Time/qmint/issues
- Q&A : https://github.com/Parallel-in-Time/qmint/discussions/categories/q-a
- Project Proposals : https://github.com/Parallel-in-Time/qmint/discussions/categories/project-proposals

## Contributors

- [Thibaut Lunet](https://github.com/tlunet)
- [Martin Schreiber](https://github.com/schreiberx)

## How to cite

```bibtex
@software{lunet2026qmint,
  author       = {Lunet, Thibaut and Schreiber, Martin},
  title        = {Parallel-in-Time/qmint},
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.21774191},
  url          = {https://doi.org/10.5281/zenodo.21774191},
}
```