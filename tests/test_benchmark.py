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


def _create_horizontally_uniform_test_data(dx, lx, ly, lz):
    """
    Field that only varies with height, so that every clean tile has (at most)
    one distinct value per height level, whereas a tile with added noise has
    a distinct value at every pixel
    """
    x_ = np.arange(0, lx, dx)
    y_ = np.arange(0, ly, dx)
    z_ = np.arange(0, lz, dx)
    ds = xr.Dataset(coords=dict(xt=x_, yt=y_, zt=z_))
    da = (0.0 * ds.xt + 0.0 * ds.yt + ds.zt / lz).transpose("xt", "yt", "zt")
    da.attrs["long_name"] = "fake water-vapour measurements"
    da.attrs["units"] = "g/kg"
    return da


class FakeCleanData:
    def __init__(self, create_fn=_create_stripy_test_data):
        da = create_fn(dx=10, lx=1000, ly=1000, lz=600)
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


def _is_clean_tile(tile, tile_size):
    # a clean tile of the horizontally uniform field has one value per height
    # level, a noisy one has a distinct value at (nearly) every pixel
    return torch.unique(tile).numel() <= tile_size


def test_n2c_model_input_is_noisy_and_target_is_clean(monkeypatch):
    """
    Noise2Clean must be trained to map the noisy field (model input) to the
    clean field (loss target)
    """
    tile_size = 32
    batch_size = 2
    with FakeCleanData(create_fn=_create_horizontally_uniform_test_data) as fp:
        dp = create_benchmarking_pipeline(
            [fp],
            model_name="n2c",
            batch_size=batch_size,
            tile_size=tile_size,
            noise_sigma=0.2,
        )
        batch = next(iter(dp))

    model = Noise2CleanDenoiserModel(n_channels=1)

    captured = {}

    class SpyNetwork(torch.nn.Module):
        def forward(self, x):
            captured["input"] = x
            return torch.zeros_like(x)

    def spy_mse_loss(y_hat, y):
        captured["target"] = y
        return torch.mean((y_hat - y) ** 2)

    monkeypatch.setattr(model, "_model", SpyNetwork())
    monkeypatch.setattr(torch.nn.functional, "mse_loss", spy_mse_loss)
    model.training_step(batch, batch_idx=0)

    model_input, model_target = captured["input"], captured["target"]
    assert model_input.shape == model_target.shape
    for i in range(batch_size):
        assert not _is_clean_tile(model_input[i], tile_size), "input must be noisy"
        assert _is_clean_tile(model_target[i], tile_size), "target must be clean"


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
