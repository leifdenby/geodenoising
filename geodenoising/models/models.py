"""
Denoising models implemented as `pytorch_lightning.LightningModule`s so that
the loss and training loop is stored with the model
"""
import pytorch_lightning as pl
from torch import nn, optim

from .ssdn.noise_network import NoiseNetwork


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


class Noise2NoiseDenoiserModel(Noise2CleanDenoiserModel):
    """
    Denoiser trained in a supervised manner with two different noisy
    realisation of the same scene, for example two microscope images of the
    same tissue sample. Apart from the data used the model is exactly same as
    for Noise2Clean and is trained in the same way using mean-squared error as
    the loss.
    """
