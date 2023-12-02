from abc import ABC, abstractmethod
from typing import Literal
import xarray as xr
import numpy as np
from scipy.special import gamma


class FootprintModel(ABC):
    @abstractmethod
    def __init__(
        self,
        ds,
        *,
        instrument_height: int,
        roughness_length: int,
        domain: xr.DataArray,
    ):
        self.ds = ds
        self.instrument_height = instrument_height
        self.roughness_length = roughness_length
        self.domain = domain
        self.calc_parameters()
        self.calc_Fx()
        self.calc_Dxy()
        self.calc_Fxy()

    @abstractmethod
    def calc_parameters(self):
        self.ds["zeta"] = self.instrument_height / self.ds["monin_obukov_lenth"]

    @abstractmethod
    def calc_Fx(self):
        raise NotImplementedError()

    @abstractmethod
    def calc_Dxy(self):
        raise NotImplementedError()

    def calc_Fxy(self):
        self.Fxy = self.Fx * self.Dxy


class HsiehFootprintModel(FootprintModel):
    def __init__(
        self,
        data,
        *,
        instrument_height: int,
        roughness_length: int,
        domain: xr.DataArray,
    ):
        super().__init__(
            data,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain=domain,
        )

    def calc_parameters(self):
        super().calc_parameters()
        self.ds["zu"] = self.instrument_height * (
            np.log(self.instrument_height / self.roughness_length)
            - 1
            + self.roughness_length / self.instrument_height
        )
        self.ds["zeta_H"] = self.ds["zu"] / self.ds["monin_obukov_lenth"]
        self.ds["D"] = xr.where(self.ds["zeta_H"] < -0.02, 0.28, 0.97)
        self.ds["D"] = xr.where(self.ds["zeta_H"] > 0.02, 2.44, self.ds["D"])
        self.ds["P"] = xr.where(self.ds["zeta_H"] < -0.02, 0.59, 1)
        self.ds["P"] = xr.where(self.ds["zeta_H"] > 0.02, 1.33, self.ds["P"])

    def calc_Dxy(self):
        sigma_y = (
            0.3
            * self.roughness_length
            * np.sqrt(self.ds["cross_wind_variance"])
            / self.ds["friction_velocity"]
        ) * ((self.domain.xx / self.roughness_length) ** 0.86)
        self.Dxy = (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(
            (-0.5) * ((self.domain.yy / sigma_y) ** 2)
        )

    def calc_Fx(self):
        self.Fx = (
            (1 / (0.41 * 0.41 * self.domain.x * self.domain.x))
            * self.ds["D"]
            * (self.ds["zu"] ** self.ds["P"])
            * (np.abs(self.ds["monin_obukov_lenth"]) ** (1 - self.ds["P"]))
            * np.exp(
                (
                    -self.ds["D"]
                    * (self.ds["zu"] ** self.ds["P"])
                    * (np.abs(self.ds["monin_obukov_lenth"]) ** (1 - self.ds["P"]))
                )
                / (0.41 * 0.41 * self.domain.x)
            )
        )


class KormannMeixnerFootprintModel(FootprintModel):
    def __init__(
        self,
        data,
        *,
        instrument_height: int,
        roughness_length: int,
        domain: xr.DataArray,
    ):
        super().__init__(
            data,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain=domain,
        )

    def calc_parameters(self):
        super().calc_parameters()
        self.ds["phi_c"] = xr.where(
            self.ds["monin_obukov_lenth"] > 0, 1 + 5 * self.ds["zeta"], 1
        )
        self.ds["phi_c"] = xr.where(
            self.ds["monin_obukov_lenth"] < 0,
            (1 - 16 * self.ds["zeta"]) ** (-0.5),
            self.ds["phi_c"],
        )
        self.ds["phi_m"] = xr.where(
            self.ds["monin_obukov_lenth"] > 0, 1 + 5 * self.ds["zeta"], 1
        )
        self.ds["phi_m"] = xr.where(
            self.ds["monin_obukov_lenth"] < 0,
            (1 - 16 * self.ds["zeta"]) ** (-0.25),
            self.ds["phi_m"],
        )
        self.ds["m"] = (
            self.ds["friction_velocity"]
            * self.ds["phi_m"]
            / (0.41 * self.ds["wind_speed"])
        )
        self.ds["n"] = xr.where(
            self.ds["monin_obukov_lenth"] > 0, 1 / (1 + 5 * self.ds["zeta"]), 1
        )
        self.ds["n"] = xr.where(
            self.ds["monin_obukov_lenth"] < 0,
            (1 - 24 * self.ds["zeta"]) / (1 - 16 * self.ds["zeta"]),
            self.ds["n"],
        )
        self.ds["U"] = self.ds["wind_speed"] / (self.instrument_height ** self.ds["m"])
        self.ds["kappa"] = (
            0.41
            * self.ds["friction_velocity"]
            * self.instrument_height
            / (self.ds["phi_c"] * self.instrument_height ** self.ds["n"])
        )
        self.ds["r"] = 2 + self.ds["m"] - self.ds["n"]
        self.ds["xi"] = (self.ds["U"] * self.instrument_height ** self.ds["r"]) / (
            self.ds["kappa"] * self.ds["r"] ** 2
        )
        self.ds["mu"] = (1 + self.ds["m"]) / self.ds["r"]

    def calc_Dxy(self):
        u_bar = (
            gamma(self.ds["mu"])
            / (gamma(1 / self.ds["r"]))
            * (
                (self.ds["kappa"] * (self.ds["r"] ** 2) / self.ds["U"])
                ** (self.ds["m"] / self.ds["r"])
            )
            * (self.ds["U"] * (self.domain.xx ** (self.ds["m"] / self.ds["r"])))
        )
        sigma_y = np.sqrt(self.ds["cross_wind_variance"]) * self.domain.xx / u_bar
        self.sigma_y = sigma_y
        self.Dxy = (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(
            (-0.5) * ((self.domain.yy / sigma_y) ** 2)
        )

    def calc_Fx(self):
        self.Fx = (
            (1 / (gamma(self.ds["mu"])))
            * (self.ds["xi"] ** self.ds["mu"])
            / (self.domain.x ** (1 + self.ds["mu"]))
            * np.exp(-self.ds["xi"] / self.domain.x)
        )


def _build_domain(
    *, domain_length: int, domain_width: int, resolution: int, time: np.ndarray
):
    x = np.linspace(1, domain_length, int(domain_length / resolution))
    y = np.linspace(-domain_width / 2, domain_width / 2, int(domain_width / resolution))
    xx, yy = np.meshgrid(x, y)
    data = np.zeros(
        (int(domain_length / resolution), int(domain_width / resolution), len(time))
    )
    da = xr.DataArray(data, dims=("x", "y", "time"))
    da = da.assign_coords(x=(("x"), x))
    da = da.assign_coords(y=(("y"), y))
    da = da.assign_coords(time=(("time"), time))
    da = da.assign_coords(xx=(("y", "x"), xx))
    da = da.assign_coords(yy=(("y", "x"), yy))
    return da


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
    domain_width: int = 500,
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
    da = _build_domain(
        domain_length=domain_length,
        domain_width=domain_width,
        resolution=resolution,
        time=time,
    )

    if method == "Hsieh":
        model = HsiehFootprintModel(
            ds,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain=da,
        )
    elif method == "Kormann & Meixner":
        model = KormannMeixnerFootprintModel(
            ds,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain=da,
        )
    return model.Fxy
