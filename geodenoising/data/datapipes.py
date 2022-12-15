from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import xarray as xr
import zen3geo  # noqa
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("open_xarray_dataset")
class OpenXarrayDatasetIterDataPipe(IterDataPipe[StreamWrapper]):
    _xr_open_method = xr.open_dataset

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[StreamWrapper]:
        for filename in self.source_datapipe:
            # getting attribute via __class__ avoids `self` being passed in
            # when we call the function
            yield StreamWrapper(self.__class__._xr_open_method(filename, **self.kwargs))

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("open_xarray_dataarray")
class OpenXarrayDataArrayIterDataPipe(OpenXarrayDatasetIterDataPipe):
    _xr_open_method = xr.open_dataarray


@functional_datapipe("normalize_xr")
class NormalizeIterDataPipe(IterDataPipe):
    """
    Normalize the data with set of fixed mean and std values

    based on https://github.com/openclimatefix/ocf_datapipes/blob/
        c317817bd207237a9cd5943f804cecb786f894a9/ocf_datapipes/transform/xarray/normalize.py
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        mean: Union[xr.Dataset, xr.DataArray, np.ndarray],
        std: Union[xr.Dataset, xr.DataArray, np.ndarray],
    ):
        self.source_datapipe = source_datapipe
        self.mean = mean
        self.std = std

    def __iter__(self) -> Union[xr.Dataset, xr.DataArray]:
        for xr_data in self.source_datapipe:
            long_name = xr_data.long_name
            xr_data = xr_data - self.mean
            xr_data = xr_data / self.std
            xr_data.attrs["long_name"] = f"Normalized {long_name}"
            xr_data.attrs["units"] = "1"
            yield xr_data


@functional_datapipe("xr_add_noise")
class AddNoiseIterDataPipe(IterDataPipe):
    """
    And normally distributed noised with mean zero and std div sigma
    """

    def __init__(self, source_datapipe: IterDataPipe[str], sigma=0.2) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.sigma = sigma

    def __iter__(self) -> Iterator[StreamWrapper]:
        for da in self.source_datapipe:
            noise = np.random.normal(scale=self.sigma, size=da.shape)
            da_noisy = da + noise
            da_noisy.attrs["long_name"] = f"{da.long_name} with added noise"
            da_noisy.attrs["units"] = da.units
            yield da_noisy

    def __len__(self) -> int:
        return len(self.source_datapipe)
