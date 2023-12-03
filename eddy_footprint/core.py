from typing import Literal
import xarray as xr
import numpy as np
from eddy_footprint.models import HsiehFootprintModel, KormannMeixnerFootprintModel


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
    domain_length: int = 1000,
    resolution: int = 1,
    method: Literal["Hsieh", "Kormann & Meixner"] = "Hsieh",
) -> xr.Dataset:
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
        )
    elif method == "Kormann & Meixner":
        model = KormannMeixnerFootprintModel(
            ds,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain_length=domain_length,
            resolution=resolution,
        )

    return model.footprints
