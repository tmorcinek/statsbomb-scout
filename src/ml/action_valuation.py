"""Module for calculating action values based on xG, goals, and xT."""
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


# def calculate_xt_values(actions: pd.DataFrame, xt_model: ExpectedThreat) -> ndarray[Any, dtype[floating[_64Bit]]]:
#     return np.nan_to_num(xt_model.rate(actions), nan=0.0)


def calculate_xt_values(actions: pd.DataFrame, xt_model: ExpectedThreat) -> ndarray[Any, dtype[floating[_64Bit]]]:
    ratings = np.zeros(len(actions))
    move_actions = get_move_actions(actions)

    grid = xt_model.xT
    # startxc, startyc = _get_cell_indexes(move_actions.start_x, move_actions.start_y, xt_model.l, xt_model.w)
    endxc, endyc = _get_cell_indexes(move_actions.end_x, move_actions.end_y, xt_model.l, xt_model.w)

    # xT_start = grid[startyc.rsub(xt_model.w - 1), startxc]
    xT_end = grid[endyc.rsub(xt_model.w - 1), endxc]

    ratings[move_actions.index] = xT_end

    return ratings
