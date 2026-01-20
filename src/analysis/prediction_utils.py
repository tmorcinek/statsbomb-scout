from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.ml.preprocessing.possessions_extraction import extract_possessions


def find_match_by_id(all_matches: List[Tuple[pd.Series, pd.DataFrame]], match_id: int) -> Tuple[pd.Series, pd.DataFrame]:
    for match, events_df in all_matches:
        if match.get('game_id') == match_id:
            return match, events_df
    raise ValueError(f"Match with game_id {match_id} not found")


def get_top_indices(predicted_values: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    sorted_indices = np.argsort(predicted_values)[::-1]
    return sorted_indices[:n] if n is not None else sorted_indices


def create_possessions_list(all_matches: List[Tuple[pd.Series, pd.DataFrame]]) -> List[Tuple[Series, dict[int, DataFrame]]]:
    return [(match, extract_possessions(match, events_df)) for match, events_df in all_matches]
