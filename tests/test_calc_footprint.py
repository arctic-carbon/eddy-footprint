from eddy_footprint import calc_footprint
import pandas as pd
import pytest
import os
import xarray as xr


@pytest.fixture(scope="module")
def data():
    cp = 1003
    fp = os.path.join(os.path.dirname(__file__), "data/flux_data_ex.csv")
    df = pd.read_csv(
        fp, parse_dates=[1], na_values="NA", delimiter=",", index_col=False, nrows=1
    )
    df["Lcalc"] = -(
        ((df["air_pressure"]) / (287 * (df["air_temperature"] + 273)))
        * cp
        * (df["u_"] ** 3)
        * (273 + df["air_temperature"])
    ) / (0.41 * 9.8 * df.H)
    return df


def test_hsieh_footprint(data):
    da = calc_footprint(
        air_pressure=data["air_pressure"],
        air_temperature=data["air_temperature"],
        friction_velocity=data["u_"],
        wind_speed=data["wind_speed"],
        cross_wind_variance=data["v_var"],
        wind_direction=data["wind_dir"],
        monin_obukhov_length=data["Lcalc"],
        time=data["datetime"],
        instrument_height=2.5,
        roughness_length=0.0206,
        method="Hsieh",
    )
    assert isinstance(da, xr.DataArray)


def test_kormann_footprint(data):
    da = calc_footprint(
        air_pressure=data["air_pressure"],
        air_temperature=data["air_temperature"],
        friction_velocity=data["u_"],
        wind_speed=data["wind_speed"],
        cross_wind_variance=data["v_var"],
        wind_direction=data["wind_dir"],
        monin_obukhov_length=data["Lcalc"],
        time=data["datetime"],
        instrument_height=2.5,
        roughness_length=0.0206,
        method="Kormann & Meixner",
    )
    assert isinstance(da, xr.DataArray)
