"""Module for calculating action values based on xG, goals, and xT."""
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from socceraction.xthreat import ExpectedThreat


def calculate_xg_value(event: pd.Series) -> float:
    if event['type_name'] != 'Shot':
        return 0.0
    shot_data = event['extra']['shot']
    if shot_data['outcome']['name'] == 'Goal':
        return 1.0
    return shot_data['statsbomb_xg']


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
