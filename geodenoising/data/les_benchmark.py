import torch
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from . import datapipes  # noqa, provide functional datapipe operations


def xr_collate_fn(samples) -> torch.Tensor:
    """
    Converts individual xarray.DataArray objects to a torch.Tensor (float32
    dtype), and stacks them all into a single torch.Tensor.
    """
    tensors = [
        torch.as_tensor(data=sample.squeeze().data.astype(dtype="float32"))
        for sample in samples
    ]
    return torch.stack(tensors=tensors)


def aggregate_mean(stream):
    # I'm sure there are better ways of doing this, but how?
    return xr.concat([item.file_obj for item in stream], dim="sample").mean()


def aggregate_std(stream):
    return xr.concat([item.file_obj for item in stream], dim="sample").std()


def create_benchmarking_pipeline(
    filepaths, model_name, n_samples=100, noise_sigma=0.2, tile_size=128, batch_size=16
):
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

    if model_name == "n2c":
        dp_clean, dp_noisy = dp_chipped_input.fork(num_instances=2)
        dp_noisy = dp_noisy.xr_add_noise(sigma=noise_sigma)
        dps = [dp_clean, dp_noisy]
    elif model_name == "n2n":
        dp_noisy1, dp_noisy2 = dp_chipped_input.fork(num_instances=2)
        dp_noisy1 = dp_noisy1.xr_add_noise(sigma=noise_sigma)
        dp_noisy2 = dp_noisy2.xr_add_noise(sigma=noise_sigma)
        dps = [dp_noisy1, dp_noisy2]
    elif model_name == "ssdn":
        dp_noisy = dp_chipped_input.xr_add_noise(sigma=noise_sigma)
        dps = [dp_noisy]
    else:
        raise NotImplementedError(model_name)

    for i, _ in enumerate(dps):
        dps[i] = (
            dps[i]
            .normalize_xr(std=std_global, mean=mean_global)
            .batch(batch_size=batch_size)
            .map(xr_collate_fn)
        )

    if len(dps) > 1:
        dp = dps[0].zip(*dps[1:])
    else:
        dp = dps[0]

    return dp.header(n_samples)
