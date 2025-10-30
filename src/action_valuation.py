"""Module for calculating action values based on xG, goals, and xT."""

import pandas as pd
import numpy as np
import socceraction.spadl as spadl
from socceraction.xthreat import ExpectedThreat
from src.xthreat import get_default_xt_model


def calculate_action_values(
    actions_df: pd.DataFrame,
    possession: pd.DataFrame,
    xt_model: ExpectedThreat
) -> pd.Series:
    """
    Calculate value for each action based on xG, goals, and xT.

    Priority:
    1. Goal = 1.0
    2. Shot (non-goal) = xG value from extra['shot']['statsbomb_xg']
    3. Other actions = delta xT (change in Expected Threat)

    Args:
        actions_df: DataFrame with SPADL actions (from convert_to_actions)
                   Must have 'original_event_id', 'type_name', 'start_x', 'start_y', 'end_x', 'end_y'
        possession: DataFrame with original StatsBomb events (from StatsBombLoader)
                   Must have 'event_id', 'type_name', 'extra' columns
        xt_model: Pre-trained xT model for calculating delta xT. If None, uses default model.

    Returns:
        pd.Series: Value for each action in actions_df
    """
    # Initialize xT model if not provided
    if xt_model is None:
        xt_model = get_default_xt_model()

    # Add type names if not present (SPADL actions have type_id, we need type_name)
    if 'type_name' not in actions_df.columns:
        actions_df = spadl.add_names(actions_df)

    values = pd.Series(0.0, index=actions_df.index)

    # Create mappings for shots and goals from 'extra' column
    xg_map = {}
    goal_events = set()

    # Extract shot information from 'extra' column
    shot_events = possession[possession['type_name'] == 'Shot'].copy()

    for _, event in shot_events.iterrows():
        event_id = event['event_id']
        extra = event.get('extra', {})

        if isinstance(extra, dict) and 'shot' in extra:
            shot_data = extra['shot']

            # Extract xG value
            if 'statsbomb_xg' in shot_data:
                xg_map[event_id] = shot_data['statsbomb_xg']

            # Check if it's a goal
            if 'outcome' in shot_data:
                outcome = shot_data['outcome']
                if isinstance(outcome, dict) and outcome.get('name') == 'Goal':
                    goal_events.add(event_id)

    # Vectorized assignment
    is_goal = actions_df['original_event_id'].isin(goal_events)
    is_shot = actions_df['type_name'] == 'shot'

    # Priority 1: Goals = 1.0
    values[is_goal] = 1.0

    # Priority 2: Shots (non-goal) = xG
    non_goal_shots = is_shot & ~is_goal
    if non_goal_shots.any():
        values[non_goal_shots] = actions_df.loc[non_goal_shots, 'original_event_id'].map(xg_map).fillna(0.0)

    # Note: Other actions (non-shots) remain at 0.0
    # Delta xT calculation can be added in the future if needed

    return values


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

