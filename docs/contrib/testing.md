# Test your changes

📜 _After doing some changes / corrections / addition in the code, you can run all the CI tests locally before any commit or PR._

## Install test dependencies

For reproducibility, it is recommended to use a dedicated environment to install all dependencies.
You can do that by running from `qmat` root folder :

```bash
python -m venv env
```

$\Rightarrow$ this will create a `env` folder in `qmat` root folder (ignored by `git`),
that you can activate using :

```bash
source ./env/bin/activate
```

> 🔔 In case you have the `base` `conda` environment as default on your computer,
> you should deactivate it before activating `env` by running `conda deactivate`.

If not already done, install all the test dependencies listed in the [pyproject.toml](../../pyproject.toml) file
under the `project.optional-dependencies` section.
Those can be installed (if not already on your system)
by running from the `qmat` root folder :

```bash
pip install -e .[test]     # install qmat locally and all test dependencies
# on MAC-OS : pip install -e ".[test]"
```

> 📣 Remember that this is the [recommended installation approach for developers](../installation.md).

## Test local changes

The first thing to do (from the root `qmint` repo) is to run :

```bash
python -c "import qmint"
```

This will trigger the [registration mechanism for the DiffOp](addDiffOp.md) at import,
and ensures that all `DiffOp` classes are correctly implemented
(in particular, overriding of the correct methods, etc ...).

Then run the full test series with :

```bash
pytest -v ./tests
```

This will test :

- the proper behavior of all `DiffOps` classes
- all the time-steppers implemented in `qmint`

## Check code coverage

Once all test pass, you may check locally coverage by running (from the `qmint` root folder) :

```bash
./test.sh
coverage combine
coverage html
```

This generates a html coverage report in `htmlcov/index.html` that you can read using your favorite web browser.

> 📣 Remember : code coverage must **stay at 100%** for a pull request to be accepted ... and the test will be reviewed to assert that they are not simple executions of your implementation 😇

## Testing notebook tutorials

All notebooks are located in the

- [`docs/basics` folder](../basics)
- [`docs/advanced` folder](../advanced)
- [`docs/features` folder](../features)

You can first check if they can be executed properly by running :

```bash
cd docs/$FOLDER
../scripts/run-sh --all
```

💡 To execute only one notebook, simply run _e.g_ :

```bash
../scripts/run-sh 2_sdc.ipynb
```

Finally, you can test all notebooks by running :

```bash
pytest ./ --nb-test-files -v
```

This will re-run each instructions in the notebooks, and compare if the generated outputs are identical to those of the locally stored notebook.