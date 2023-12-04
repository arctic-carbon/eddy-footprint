from typing import Literal
import xarray as xr
import numpy as np
from eddy_footprint.models import HsiehFootprintModel, KormannMeixnerFootprintModel
from typing import Optional


def calc_footprint(
    *,
    air_pressure: np.ndarray,
    air_temperature: np.ndarray,
    friction_velocity: np.ndarray,
    wind_speed: np.ndarray,
    cross_wind_variance: np.ndarray,
    wind_direction: np.ndarray,
    monin_obukov_lenth: np.ndarray,
    time: np.ndarray,
    instrument_height: float,
    roughness_length: float,
    domain_length: Optional[int] = 1000,
    resolution: Optional[int] = 5,
    workers: Optional[int] = 1,
    method: Optional[Literal["Hsieh", "Kormann & Meixner"]] = "Hsieh",
) -> xr.DataArray:
    """Create a dataset with footprint influences from flux tower measurements.

    .. warning::
        This function is experimental and its signature may change.

    Parameters
    ----------
    air_pressure : np.ndarray
        Array with measurement of air pressure in units.
    air_temperature : np.ndarray
    friction_velocity : np.ndarray
    instrument_height : float
    roughness_length : float
    domain_length : int, optional
    resolution : int, optional
    workers : int, optional
        Number of workers to use for parallel processing during interpolation step.
        If -1 is given all CPU threads are used. Default: 1.

    Returns
    -------
    da: xarray.DataArray
        DataArray with footprints of influence.
    """
    ds = xr.Dataset()
    ds["air_pressure"] = xr.DataArray(
        data=air_pressure, dims=["time"], coords=dict(time=time)
    )
    ds["air_temperature"] = xr.DataArray(
        data=air_temperature, dims=["time"], coords=dict(time=time)
    )
    ds["friction_velocity"] = xr.DataArray(
        data=friction_velocity, dims=["time"], coords=dict(time=time)
    )
    ds["wind_speed"] = xr.DataArray(
        data=wind_speed, dims=["time"], coords=dict(time=time)
    )
    ds["cross_wind_variance"] = xr.DataArray(
        data=cross_wind_variance, dims=["time"], coords=dict(time=time)
    )
    ds["wind_direction"] = xr.DataArray(
        data=wind_direction, dims=["time"], coords=dict(time=time)
    )
    ds["monin_obukov_lenth"] = xr.DataArray(
        data=monin_obukov_lenth, dims=["time"], coords=dict(time=time)
    )

    if method == "Hsieh":
        model = HsiehFootprintModel(
            ds,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain_length=domain_length,
            resolution=resolution,
            workers=workers,
        )
    elif method == "Kormann & Meixner":
        model = KormannMeixnerFootprintModel(
            ds,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain_length=domain_length,
            resolution=resolution,
            workers=workers,
        )

    return model.footprints
