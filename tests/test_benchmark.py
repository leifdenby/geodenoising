"""
tests for creating dataset for benchmark and running benchmark training
"""
import tempfile

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import xarray as xr

from geodenoising.benchmark import BenchmarkingDataModule, Noise2CleanDenoiserModel
from geodenoising.data.les_benchmark import create_benchmarking_pipeline

MODELS = "n2c n2n ssdn".split()


def _create_stripy_test_data(dx, lx, ly, lz):
    x_ = np.arange(0, lx, dx)
    y_ = np.arange(0, ly, dx)
    z_ = np.arange(0, lz, dx)
    ds = xr.Dataset(coords=dict(xt=x_, yt=y_, zt=z_))
    da = np.sin((ds.xt * ds.yt) * ds.zt)
    da.attrs["long_name"] = "fake water-vapour measurements"
    da.attrs["units"] = "g/kg"
    return da


class FakeCleanData:
    def __init__(self):
        da = _create_stripy_test_data(dx=10, lx=1000, ly=1000, lz=600)
        self.tfile = tempfile.NamedTemporaryFile(suffix=".nc")  # noqa
        da.to_netcdf(self.tfile.name)
        da.close()

    def __enter__(self):
        return self.tfile.name

    def __exit__(self, *arg, **kwargs):
        del self.tfile


@pytest.mark.parametrize("model_name", MODELS)
def test_benchmark_pipeline(model_name):
    with FakeCleanData() as fp:
        tile_size = 32
        batch_size = 2
        dp = create_benchmarking_pipeline(
            [fp], model_name=model_name, batch_size=batch_size, tile_size=tile_size
        )
        batch = next(iter(dp))
        assert len(batch) == batch_size

        item = batch[0]
        if model_name == "ssdn":
            assert torch.is_tensor(item)
        elif model_name in ["n2n", "n2c"]:
            assert len(item)
            assert all(torch.is_tensor(v) for v in item)
        else:
            raise NotImplementedError(model_name)


def test_train():
    with FakeCleanData() as fp:
        model_name = "n2c"
        dm = BenchmarkingDataModule(
            model_name=model_name, n_samples=5, fps_train=[fp], fps_test=[fp]
        )
        model = Noise2CleanDenoiserModel(n_channels=dm.n_channels)

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
        )

        trainer.fit(model=model, train_dataloaders=dm)
