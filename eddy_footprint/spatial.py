import numpy as np
import xarray as xr
from scipy.spatial import KDTree


def rotate_domain(da, *, wind_direction):
    rot = -(wind_direction) * np.pi / 180
    x = da.xx.data * np.cos(rot) + da.yy.data * np.sin(rot)
    y = -da.xx.data * np.sin(rot) + da.yy.data * np.cos(rot)
    da = da.assign_coords(x=(("y", "x"), x))
    da = da.assign_coords(y=(("y", "x"), y))
    da = da.drop_vars(["xx", "yy"])
    return da


def build_domain(*, domain_length: int, resolution: int, time: np.ndarray):
    x = np.linspace(0, domain_length, int(domain_length / resolution))
    y = np.linspace(
        -domain_length / 2, domain_length / 2, int(domain_length / resolution)
    )
    xx, yy = np.meshgrid(x, y)
    data = np.zeros(
        (
            int(domain_length / resolution),
            int(domain_length / resolution),
            len(time),
        )
    )
    da = xr.DataArray(data, dims=("x", "y", "time"))
    da = da.assign_coords(x=(("x"), x))
    da = da.assign_coords(y=(("y"), y))
    da = da.assign_coords(time=(("time"), time))
    da = da.assign_coords(xx=(("y", "x"), xx))
    da = da.assign_coords(yy=(("y", "x"), yy))
    return da


def resample(da, *, query_points, output_shape, workers):
    points = np.array((da.x.data.flatten(), da.y.data.flatten())).transpose()
    tree = KDTree(points)
    d, ind = tree.query(query_points, k=4, workers=workers)
    w = 1.0 / d**2
    output_points = np.sum(w * da.data.flatten()[ind], axis=1) / np.sum(w, axis=1)
    output_points.shape = output_shape
    return output_points


def normalize_domain(
    da, *, wind_direction, query_points, template_xx, template_x, template_y, workers
):
    da = rotate_domain(da, wind_direction=wind_direction)
    da = da.transpose("x", "y")
    output_points = resample(
        da, query_points=query_points, output_shape=template_xx.shape, workers=workers
    )
    output_ds = xr.DataArray(data=output_points, dims=("x", "y"))
    output_ds = output_ds.assign_coords(x=template_x)
    output_ds = output_ds.assign_coords(y=template_y)
    return output_ds


def build_template(*, domain_length, resolution):
    template_x = np.linspace(
        -domain_length, domain_length, int(domain_length / resolution) * 2
    )
    template_y = np.linspace(
        -domain_length, domain_length, int(domain_length / resolution) * 2
    )
    template_xx, template_yy = np.meshgrid(template_x, template_y, indexing="xy")
    query_points = np.array((template_xx.flatten(), template_yy.flatten())).transpose()
    return query_points, template_xx, template_yy, template_x, template_y
