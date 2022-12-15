"""
Routines for benchmarking all denoising techniques using clean-sample data with
added noise
"""
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader

from .data.les_benchmark import (
    create_benchmarking_pipeline,
    create_training_benchmark_pipeline,
)
from .models import Noise2CleanDenoiserModel, Noise2NoiseDenoiserModel


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


def _create_benchmark_model_and_datamodule(model_name):
    dm = BenchmarkingDataModule(model_name=model_name, n_dataloader_workers=6)
    if model_name == "n2c":
        model = Noise2CleanDenoiserModel(n_channels=dm.n_channels)
    elif model_name == "n2n":
        model = Noise2NoiseDenoiserModel(n_channels=dm.n_channels)
    else:
        raise NotImplementedError(model_name)

    return model, dm


def main():
    import argparse  # noqa

    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_name")
    args = argparser.parse_args()

    model, dm = _create_benchmark_model_and_datamodule(model_name=args.model_name)

    assert torch.cuda.is_available()

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model=model, train_dataloaders=dm)


if __name__ == "__main__":
    main()
