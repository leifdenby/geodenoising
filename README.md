# geodenoising

This repository will contain tools for applying denoising tools in geophysical
applications, firstly applied to denoising LIDAR observations of water vapour
concentration below cloud in Earth's atmosphere.

## Running the tests

The dependencies in `setup.cfg` are unpinned, and current releases of
`torchdata` (>= 0.10) no longer contain `torchdata.datapipes`, which this
package uses. Use the pinned environment in `requirements/test-2022-12.txt`
(Python 3.10, all packages as of 2022-12-17):

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements/test-2022-12.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest
```
