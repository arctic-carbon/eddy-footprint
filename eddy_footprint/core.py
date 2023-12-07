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
    monin_obukhov_length: np.ndarray,
    time: np.ndarray,
    instrument_height: float,
    roughness_length: float,
    domain_length: Optional[int] = 1000,
    resolution: Optional[int] = 5,
    workers: Optional[int] = 1,
    method: Optional[Literal["Hsieh", "Kormann & Meixner"]] = "Hsieh",
) -> xr.DataArray:
    """Create a dataset with footprint influences from eddy covariance measurements.

    .. warning::
        This function is experimental and its signature may change.

    Parameters
    ----------
    air_pressure : np.ndarray
        Array with measurement of air pressure in units Pa.
    air_temperature : np.ndarray
        Array with measurments of air temperature in degrees K.
    friction_velocity : np.ndarray
        Array with measurements of friction veloicty (u*) in meters per second.
    wind_speed : np.ndarray
        Array with horizontal (u) wind velocity in meters per second.
    cross_wind_variance : np.ndarray
        Array with cross wind (v) variance in meters^2 per second^2.
    wind_direction : np.ndarray
        Array with down wind direction in degrees, where N/S is at 0/360 degrees.
    monin_obukhov_length : np.ndarray
        Array with Monin-Obukhov length in meters.
    time : np.ndarray
        Array with time, a datetime object, filename, or other unique identifier.
        time will serve as the third dimension.
        time need not be continuous or regular, but cannot have duplicates.
    instrument_height : float
        Constant for the instrument (sonic anemometer) height in meters above ground.
    roughness_length : float
        Constant for the site roughness length (z_not) in meters.
    domain_length : int, optional
        Integer for the domain length in meters.
        The extent in downwind and crosswind directions used in footprint calculations.
        The x and y dimensions of the output xarray are twice the domain length,
        with the eddy covaiance tower located at (0,0).
        Default: 1000.
    resolution : int, optional
        Integer for the resolution in meters used in the footprint calculations
        and x and y dimensions. Default: 5.
    workers : int, optional
        Number of workers to use for parallel processing during interpolation step.
        If -1 is given all CPU threads are used. Default: 1.
    method : ``Hsieh`` or ``Kormann & Meixner``, optional
        The footprint model method to use, either Hsieh or Kormann & Meixner.
        Default: Hsieh.


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
    ds["monin_obukhov_length"] = xr.DataArray(
        data=monin_obukhov_length, dims=["time"], coords=dict(time=time)
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
