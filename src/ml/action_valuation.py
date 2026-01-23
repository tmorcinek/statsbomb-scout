"""Module for calculating action values based on xG, goals, and xT."""
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from socceraction.xthreat import ExpectedThreat


def calculate_xg_values(events: pd.DataFrame) -> pd.Series:
    xg_values = pd.Series(0.0, index=events.index)
    shots = events[events["type_name"] == "Shot"]

    if shots.empty:
        return xg_values

    def extract_xg(extra):
        return extra.get("shot", {}).get("statsbomb_xg", 0.0)

    xg_values.loc[shots.index] = shots["extra"].map(extract_xg)
    return xg_values


def calculate_xt_values(actions: pd.DataFrame, xt_model: ExpectedThreat) -> ndarray[Any, dtype[floating[_64Bit]]]:
    return np.nan_to_num(xt_model.rate(actions), nan=0.0)
