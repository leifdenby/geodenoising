import numpy as np
import torch
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from ..models.n2v.input import manipulate as manipulate_input
from . import datapipes  # noqa, provide functional datapipe operations


def xr_to_tensor(da) -> torch.Tensor:
    data = da.squeeze().data.astype(dtype="float32")
    tensor = torch.as_tensor(data=data)
    return tensor


def xr_collate_fn(samples) -> torch.Tensor:
    """
    Converts individual xarray.DataArray objects to a torch.Tensor (float32
    dtype), and stacks them all into a single torch.Tensor.
    """
    tensors = []

    for sample in samples:
        data = sample.squeeze().data.astype(dtype="float32")
        if len(data.shape) == 2:
            # add channel axis if we've only got one channel
            data = data[np.newaxis, ...]
        tensor = torch.as_tensor(data=data)
        tensors.append(tensor)

    return torch.stack(tensors=tensors)


def add_channel_dim(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor[:, None, :, :]
    return tensor


def aggregate_mean(stream):
    # I'm sure there are better ways of doing this, but how?
    return xr.concat([item.file_obj for item in stream], dim="sample").mean()


def aggregate_std(stream):
    return xr.concat([item.file_obj for item in stream], dim="sample").std()


def _create_source_chips_datapipe(filepaths, tile_size):
    dp_src = IterableWrapper(filepaths).open_xarray_dataarray()

    dp_src_x, dp_src_y = dp_src.fork(num_instances=2)

    mean_global = next(iter(dp_src.zip().map(aggregate_mean))).item()
    std_global = next(iter(dp_src.zip().map(aggregate_std))).item()

    dp_chipped_input_x = dp_src_x.slice_with_xbatcher(
        input_dims=dict(xt=tile_size, yt=1, zt=tile_size)
    )
    dp_chipped_input_y = dp_src_y.slice_with_xbatcher(
        input_dims=dict(yt=tile_size, xt=1, zt=tile_size)
    )
    dp_chipped_input = dp_chipped_input_x.mux(dp_chipped_input_y).shuffle()

    return dp_chipped_input, mean_global, std_global


def _create_n2c_benchmarking_pipeline(
    dp_clean_chips, noise_sigma, batch_size, mean_global, std_global
):
    dp_clean, dp_noisy = dp_clean_chips.fork(num_instances=2)
    dp_noisy = dp_noisy.xr_add_noise(sigma=noise_sigma)

    def chips_to_batch_tensor(dp):
        return (
            dp.normalize_xr(std=std_global, mean=mean_global)
            .map(xr_to_tensor)
            .batch(batch_size=batch_size)
            .map(torch.stack)
            .map(add_channel_dim)
        )

    dp_clean = chips_to_batch_tensor(dp_clean)
    dp_noisy = chips_to_batch_tensor(dp_noisy)

    return dp_clean.zip(dp_noisy)


def _create_n2n_benchmarking_pipeline(
    dp_clean_chips, noise_sigma, batch_size, mean_global, std_global
):
    dp_noisy1, dp_noisy2 = dp_clean_chips.fork(num_instances=2)

    def clean_chips_to_noisy_batch_tensor(dp):
        return (
            dp.xr_add_noise(sigma=noise_sigma)
            .normalize_xr(std=std_global, mean=mean_global)
            .map(xr_to_tensor)
            .batch(batch_size=batch_size)
            .map(torch.stack)
            .map(add_channel_dim)
        )

    dp_noisy1 = clean_chips_to_noisy_batch_tensor(dp_noisy1)
    dp_noisy2 = clean_chips_to_noisy_batch_tensor(dp_noisy2)

    return dp_noisy1.zip(dp_noisy2)


def _create_ssdn_benchmarking_pipeline(
    dp_clean_chips, noise_sigma, batch_size, mean_global, std_global
):
    return (
        dp_clean_chips.xr_add_noise(sigma=noise_sigma)
        .normalize_xr(std=std_global, mean=mean_global)
        .map(xr_to_tensor)
        .batch(batch_size=batch_size)
        .map(torch.stack)
        .map(add_channel_dim)
    )


def _create_n2v_benchmarking_pipeline(
    dp_clean_chips, noise_sigma, batch_size, mean_global, std_global
):
    dp_noisy = (
        dp_clean_chips.xr_add_noise(sigma=noise_sigma)
        .normalize_xr(std=std_global, mean=mean_global)
        .map(xr_to_tensor)
    )

    dp_manipulated, dp_manipulated_coords = (
        dp_chips.map(xr_to_tensor).map(manipulate).unzip(sequence_length=2)
    )

    dp_noisy, dp_noisy_manipulated = dp_noisy.fork(num_instances=2)

    dp_noisy_manipulated = dp_noisy_manipulated.map(manipulate_input)
    dp_noisy_manipulated, dp_noisy_manipulated_coords = dp_noisy_manipulated.unzip(
        sequence_length=2
    )

    batch_and_stack = lambda dp: dp.batch(batch_size=batch_size).map(torch.stack)

    dp = (
        batch_and_stack(dp_noisy_manipulated)
        .zip(batch_and_stack(dp_noisy), batch_and_stack(dp_noisy_manipulated_coords))
        .map(add_channel_dim)
    )
    return dp


def create_benchmarking_pipeline(
    filepaths, model_name, n_samples=100, noise_sigma=0.2, tile_size=128, batch_size=16
):
    dp_chipped_input, mean_global, std_global = _create_source_chips_datapipe(
        filepaths=filepaths, tile_size=tile_size
    )

    if model_name == "n2c":
        dp = _create_n2c_benchmarking_pipeline(
            dp_clean_chips=dp_chipped_input,
            noise_sigma=noise_sigma,
            batch_size=batch_size,
            mean_global=mean_global,
            std_global=std_global,
        )
    elif model_name == "n2n":
        dp = _create_n2n_benchmarking_pipeline(
            dp_clean_chips=dp_chipped_input,
            noise_sigma=noise_sigma,
            batch_size=batch_size,
            mean_global=mean_global,
            std_global=std_global,
        )
    elif model_name == "ssdn":
        dp = _create_ssdn_benchmarking_pipeline(
            dp_clean_chips=dp_chipped_input,
            noise_sigma=noise_sigma,
            batch_size=batch_size,
            mean_global=mean_global,
            std_global=std_global,
        )
    elif model_name == "n2v":
        dp = _create_n2v_benchmarking_pipeline(
            dp_clean_chips=dp_chipped_input,
            noise_sigma=noise_sigma,
            batch_size=batch_size,
            mean_global=mean_global,
            std_global=std_global,
        )
    else:
        raise NotImplementedError(model_name)

    return dp.header(n_samples)


def create_training_benchmark_pipeline(
    filepaths,
    model_name,
    n_samples=100,
    train_val_split=0.9,
    tile_size=128,
    batch_size=16,
    seed=42,
    noise_sigma=0.2,
):
    dp = create_benchmarking_pipeline(
        filepaths=filepaths,
        model_name=model_name,
        tile_size=tile_size,
        batch_size=batch_size,
        n_samples=n_samples,
        noise_sigma=noise_sigma,
    )
    return dp.random_split(
        weights=dict(train=train_val_split, val=1.0 - train_val_split),
        seed=seed,
        total_length=n_samples,
    )
