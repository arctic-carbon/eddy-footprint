from abc import ABC, abstractmethod

import xarray as xr
import numpy as np
from scipy.special import gamma
from eddy_footprint.spatial import build_domain, build_template, normalize_domain


class FootprintModel(ABC):
    @abstractmethod
    def __init__(
        self,
        ds,
        *,
        instrument_height: int,
        roughness_length: int,
        domain_length: int,
        resolution: int,
        workers: int,
    ):
        self.ds = ds
        self.instrument_height = instrument_height
        self.roughness_length = roughness_length
        self.domain = build_domain(
            domain_length=domain_length, resolution=resolution, time=self.ds.time.data
        )
        self.calc_parameters()
        self.calc_Fx()
        self.calc_Dxy()
        self.calc_Fxy()
        query_points, template_xx, _, template_x, template_y = build_template(
            domain_length=domain_length,
            resolution=resolution,
        )
        datasets = []
        for timestep in self.ds.time:
            timestep_ds = normalize_domain(
                self.ds["Fxy"].sel(time=timestep),
                wind_direction=ds["wind_direction"].sel(time=timestep).data,
                query_points=query_points,
                template_xx=template_xx,
                template_x=template_x,
                template_y=template_y,
                workers=workers,
            )
            timestep_ds = timestep_ds.expand_dims(dim={"time": [timestep.values]})
            datasets.append(timestep_ds)
        self.footprints = xr.concat(datasets, dim="time")

    @abstractmethod
    def calc_parameters(self):
        self.ds["zeta"] = self.instrument_height / self.ds["monin_obukhov_length"]

    @abstractmethod
    def calc_Fx(self):
        raise NotImplementedError()

    @abstractmethod
    def calc_Dxy(self):
        raise NotImplementedError()

    def calc_Fxy(self):
        self.ds["Fxy"] = self.Fx * self.Dxy


class HsiehFootprintModel(FootprintModel):
    def __init__(
        self,
        data,
        *,
        instrument_height: int,
        roughness_length: int,
        domain_length: int,
        resolution: int,
        workers: int,
    ):
        super().__init__(
            data,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain_length=domain_length,
            resolution=resolution,
            workers=workers,
        )

    def calc_parameters(self):
        super().calc_parameters()
        self.ds["zu"] = self.instrument_height * (
            np.log(self.instrument_height / self.roughness_length)
            - 1
            + self.roughness_length / self.instrument_height
        )
        self.ds["zeta_H"] = self.ds["zu"] / self.ds["monin_obukhov_length"]
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
            * (np.abs(self.ds["monin_obukhov_length"]) ** (1 - self.ds["P"]))
            * np.exp(
                (
                    -self.ds["D"]
                    * (self.ds["zu"] ** self.ds["P"])
                    * (np.abs(self.ds["monin_obukhov_length"]) ** (1 - self.ds["P"]))
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
        domain_length: int,
        resolution: int,
        workers: int,
    ):
        super().__init__(
            data,
            instrument_height=instrument_height,
            roughness_length=roughness_length,
            domain_length=domain_length,
            resolution=resolution,
            workers=workers,
        )

    def calc_parameters(self):
        super().calc_parameters()
        self.ds["phi_c"] = xr.where(
            self.ds["monin_obukhov_length"] > 0, 1 + 5 * self.ds["zeta"], 1
        )
        self.ds["phi_c"] = xr.where(
            self.ds["monin_obukhov_length"] < 0,
            (1 - 16 * self.ds["zeta"]) ** (-0.5),
            self.ds["phi_c"],
        )
        self.ds["phi_m"] = xr.where(
            self.ds["monin_obukhov_length"] > 0, 1 + 5 * self.ds["zeta"], 1
        )
        self.ds["phi_m"] = xr.where(
            self.ds["monin_obukhov_length"] < 0,
            (1 - 16 * self.ds["zeta"]) ** (-0.25),
            self.ds["phi_m"],
        )
        self.ds["m"] = (
            self.ds["friction_velocity"]
            * self.ds["phi_m"]
            / (0.41 * self.ds["wind_speed"])
        )
        self.ds["n"] = xr.where(
            self.ds["monin_obukhov_length"] > 0, 1 / (1 + 5 * self.ds["zeta"]), 1
        )
        self.ds["n"] = xr.where(
            self.ds["monin_obukhov_length"] < 0,
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
