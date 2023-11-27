import pandas as pd
from typing import Literal


def calc_footprint(
    data: pd.DataFrame,
    *,
    domain_length: int = 1000,
    resolution: int = 1,
    method: Literal["Hsieh", "Kormann & Meixner"],
    instrument_height: float,
    roughness_length: float,
):
    pass
