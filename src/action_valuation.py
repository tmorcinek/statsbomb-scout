"""Module for calculating action values based on xG, goals, and xT."""

import pandas as pd
import numpy as np
import socceraction.spadl as spadl
from socceraction.xthreat import ExpectedThreat
from src.xthreat import get_default_xt_model

def calculate_post_event_value(action: pd.Series) -> float:
    if action['type_name'] != 'Shot':
        return 0.0
    shot_data = action['extra']['shot']
    if shot_data['outcome']['name'] == 'Goal':
        return 1.0
    return shot_data['statsbomb_xg']




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

