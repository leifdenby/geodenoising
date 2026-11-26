# geodenoising

This repository will contain tools for applying denoising tools in geophysical
applications, firstly applied to denoising LIDAR observations of water vapour
concentration below cloud in Earth's atmosphere.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). The
dependencies are pinned to the 2022-12 stack (Python 3.10, torch 1.13,
torchdata 0.5, pytorch-lightning 1.8); current torchdata (>= 0.10) no longer
contains `torchdata.datapipes`, which this package uses. `uv.lock` holds the
exact versions, and torch is installed as the CPU-only build from the PyTorch
package index.

```bash
uv sync
```

This creates `.venv/` with Python 3.10, geodenoising (editable), and the
`test` and `dev` dependency groups (pytest, pre-commit, ipython, jupyterlab,
nbval). For only the runtime and test dependencies, use
`uv sync --no-default-groups --group test`.

To run the pre-commit hooks on each commit:

```bash
uv run pre-commit install
```

## Running the tests

```bash
uv run pytest
```

`requirements/test-2022-12.txt` records the same 2022-12 environment as a
plain requirements file (with the PyPI CUDA build of torch); see its header
for how to install from it without `uv sync`.
