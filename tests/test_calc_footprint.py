from eddy_footprint import calc_footprint
import pandas as pd
import pytest
import os
import xarray as xr
import rioxarray


@pytest.fixture(scope="module")
def data():
    fp = os.path.join(os.path.dirname(__file__), "data/flux_data_ex.csv")
    return pd.read_csv(
        fp, parse_dates=[1], na_values="NA", delimiter=",", index_col=False, nrows=1
    )


@pytest.fixture(scope="module")
def expected_hsieh():
    fp = os.path.join(os.path.dirname(__file__), "data/2019-07-11T000000_H.tif")
    return rioxarray.open_rasterio(fp)


@pytest.fixture(scope="module")
def expected_kormann():
    fp = os.path.join(os.path.dirname(__file__), "data/2019-07-11T000000_KM.tif")
    return rioxarray.open_rasterio(fp)


def test_hsieh_footprint(data, expected_hsieh):
    ds = calc_footprint(
        data=data, instrument_height=2.5, roughness_length=0.0206, method="Hsieh"
    )
    xr.testing.assert_allclose(ds, expected_hsieh)


def test_kormann_footprint(data, expected_kormann):
    ds = calc_footprint(
        data=data,
        instrument_height=2.5,
        roughness_length=0.0206,
        method="Kormann & Meixner",
    )
    xr.testing.assert_allclose(ds, expected_kormann)
