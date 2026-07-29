# PES_DL

Deep learning models for constructing potential energy surfaces from a sparse
set of ab initio points, benchmarked against classical interpolation (cubic
spline, Gaussian process).

## Install

```bash
git clone https://github.com/abatista-planas/PES_DL.git
cd PES_DL
pip install -e .
```

Needs Python 3.8+ and PyTorch. For the dev tools (black, ruff, isort, pytest,
pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

## Tests

```bash
pytest
```

## License

GPL-3.0, see [LICENSE](LICENSE).
