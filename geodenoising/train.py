from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn, optim
from torch.utils.data import DataLoader

from .data.les_benchmark import (
    create_benchmarking_pipeline,
    create_training_benchmark_pipeline,
)
from .models.ssdn.noise_network import NoiseNetwork


class Noise2CleanDenoiserModel(pl.LightningModule):
    """
    Denoiser trained in a supervised manner with (noisy, clean) pairs
    Use mean-squared error as loss
    """

    def __init__(self, n_channels):
        super().__init__()
        self._model = NoiseNetwork(
            in_channels=n_channels, out_channels=n_channels, blindspot=False
        )

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        y_hat = self._model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        y_hat = self._model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self._model(x)


class Denoiser(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        if model_name == "n2c":
            pass
        else:
            raise NotImplementedError(model_name)


class BenchmarkingDataModule(pl.LightningDataModule):
    def __init__(self, model_name, n_samples=100, n_dataloader_workers=6):
        super().__init__()
        self.model_name = model_name
        self.n_samples = n_samples
        self.noise_sigma = 0.2
        self.tile_size = 32
        self.batch_size = 8
        self.train_val_split = 0.9

        self.n_channels = 1

        self.dp_train = None
        self.dp_val = None
        self.dp_test = None

        self.n_dataloader_workers = n_dataloader_workers

    def setup(self, stage):
        kwargs = dict(
            model_name=self.model_name,
            n_samples=self.n_samples,
            noise_sigma=self.noise_sigma,
            tile_size=self.tile_size,
            batch_size=self.batch_size,
        )
        fp_root = Path(__file__).parent.parent / "datasets_prep"
        if stage == "fit" or stage is None:
            filepaths = [fp_root / "rico.no_shear_br0.05.qv.tn6.nc"]
            self.dp_train, self.dp_val = create_training_benchmark_pipeline(
                filepaths, train_val_split=self.train_val_split, **kwargs
            )
        if stage == "test" or stage is None:
            filepaths = [fp_root / "rico.no_shear_br0.05.qv.tn7.nc"]
            self.dp_test = create_benchmarking_pipeline(filepaths, **kwargs)

    def train_dataloader(self):
        # batching already done in datapipe
        return DataLoader(
            self.dp_train, batch_size=None, num_workers=self.n_dataloader_workers
        )

    def val_dataloader(self):
        # batching already done in datapipe
        return DataLoader(
            self.dp_val, batch_size=None, num_workers=self.n_dataloader_workers
        )

    def test_dataloader(self):
        # batching already done in datapipe
        return DataLoader(
            self.dp_test, batch_size=None, num_workers=self.n_dataloader_workers
        )


if __name__ == "__main__":
    dm = BenchmarkingDataModule(model_name="n2c", n_dataloader_workers=6)
    model = Noise2CleanDenoiserModel(n_channels=dm.n_channels)

    assert torch.cuda.is_available()

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model=model, train_dataloaders=dm)
