from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from socceraction.xthreat import ExpectedThreat, get_move_actions, _get_cell_indexes


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
    ratings = np.zeros(len(actions))
    move_actions = get_move_actions(actions)
    grid = xt_model.xT

    # Calculate xT for move_actions based on end position
    if len(move_actions) > 0:
        endxc, endyc = _get_cell_indexes(move_actions.end_x, move_actions.end_y, xt_model.l, xt_model.w)
        xT_end = grid[endyc.rsub(xt_model.w - 1), endxc]
        move_positions = actions.index.get_indexer(move_actions.index)
        ratings[move_positions] = xT_end

    # Calculate xT for non-move actions based on start position
    non_move_actions = actions[~actions.index.isin(move_actions.index)]
    if len(non_move_actions) > 0:
        startxc, startyc = _get_cell_indexes(non_move_actions.start_x, non_move_actions.start_y, xt_model.l, xt_model.w)
        xT_start = grid[startyc.rsub(xt_model.w - 1), startxc]
        non_move_positions = actions.index.get_indexer(non_move_actions.index)
        ratings[non_move_positions] = xT_start

    return ratings


def get_actions_value(actions_df: pd.DataFrame) -> float:
    total_xg = actions_df['xG'].sum()
    xt = actions_df['xT'].iloc[-1]
    return max(total_xg, xt)
