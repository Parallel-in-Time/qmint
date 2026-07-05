
# Installation

`qmat` is a lightweight package to generate time-integration coefficients, which can be use as core dependency in any solver for Initial Value Problems (IVP).
It depends only on [`numpy`](https://numpy.org) & [`scipy`](https://scipy.org), and can be installed from [`pypi`](https://pypi.org/) using :

```bash
pip install qmat
```

`qmint` is an prototyping package implementing different kind of IVP solvers, that can be used for learning or experiments.
It has additional dependencies compared to `qmat` ([`matplotlib`](https://matplotlib.org), ...) and can be installed from [`pypi`](https://pypi.org/) using :

```bash
pip install qmint
```

> 📣 Installing `qmint` will automatically install `qmat` if it's not already installed on your Python environment.


## Install using conda

Currently, no version is distributed on conda-forge. However using `pip` from `conda` will install `qmat`/`qmint` in your conda environment.

If you are using a `environment.yml` file with conda, then you can add it as a dependency like this $:$

```yaml
name: yourEnv
channels:
  - conda-forge
  - defaults
dependencies:
  ...
  - pip
  - pip:
    - qmat # or qmint
```

## Install `qmint` from source

In case you want the latest revision (or a specific branch), you can directly clone the sources from `github` :

```bash
git clone https://github.com/Parallel-in-Time/qmint.git
```

If you **want to use the package only**, simply use the `pip` local installer directly :

```bash
cd qmint     # go into the local git repo
pip install .
```

For **developers who want to contribute**, recommended approach is to install
the package in _editable mode_ :

```bash
cd qmint     # go into the local git repo (if not already there)
pip install -e .[test]    # on MAC-OS : pip install -e ".[test]"
```

This will link your python installation to your local `qmint` folder,
hence all your modifications will be taken into account at each new import of `qmint`.

> 🔔 Some IDEs also modify the `PYTHONPATH` to include the `qmint` root folder, which you can also do manually if you prefer.
