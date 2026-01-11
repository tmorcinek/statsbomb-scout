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


def extract_label_from_last_action(actions_df: pd.DataFrame) -> np.ndarray:
    """
    Extract label from the last action's pre-calculated value.

    Assumes actions_df has a 'value' column calculated by calculate_action_values().

    Args:
        actions_df: DataFrame with actions and 'value' column

    Returns:
        np.ndarray: Single-element array with the last action's value
    """
    if 'value' not in actions_df.columns:
        raise ValueError("actions_df must have 'value' column. Call calculate_action_values() first.")

    return np.array([actions_df['value'].iloc[-1]], dtype=np.float32)
