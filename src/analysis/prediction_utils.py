from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.analysis.game_utils import game_summary
from src.analysis.visualization import plot_multiple_possessions, _title_with_value
from src.ml.preprocessing.possessions_extraction import extract_possessions, tail_dataframes_to_sequence_length


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


def matches_info_df(matches: List[Tuple[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'match_id': match.get('match_id', match.get('game_id', 'N/A')),
            'summary': game_summary(match),
            'events_count': len(events_df),
        }
        for match, events_df in matches
    ])


def print_top_predictions(predicted_values: np.ndarray, possessions: np.ndarray, matches: Optional[np.ndarray], top_n: int,
                          attention_weights: Optional[np.ndarray] = None) -> None:
    top_indices = get_top_indices(predicted_values, top_n)

    print(f"\nTop {top_n} sequences with highest predicted values:\n")
    for rank, idx in enumerate(top_indices):
        seq_value = predicted_values[idx]
        possession_id = possessions[idx]

        if matches is not None:
            print(f"{rank + 1}. Match ID = {matches[idx]}, Possession ID = {possession_id}: Value = {seq_value:.3f}")
        else:
            print(f"{rank + 1}. Possession id = {possession_id}: Value = {seq_value:.3f}")
        if attention_weights is not None:
            seq_weights = attention_weights[idx]
            max_weight_idx = np.argmax(seq_weights)
            print(f"   Attention weights: {[f'{w:.3f}' for w in seq_weights]}")
            print(f"   Most important action: position {max_weight_idx} (weight: {seq_weights[max_weight_idx]:.3f})")
        print()


def normalize_predictions(predictions) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(predictions, dict):
        return predictions['value'].flatten(), predictions['attention_weights']
    return predictions.flatten(), None


def visualize_top_possessions(selected_match: Series, selected_events: DataFrame, p_match: np.ndarray,
                              predicted_values: np.ndarray, attention_weights: Optional[np.ndarray],
                              top_n: int, sequence_length: int):
    from src.analysis.visualization import plot_multiple_possessions, _title_with_value
    from src.ml.preprocessing.possessions_extraction import tail_dataframes_to_sequence_length
    import matplotlib.pyplot as plt

    top_indices = get_top_indices(predicted_values, top_n)

    possessions_dict = extract_possessions(selected_match, selected_events)
    top_possessions = [possessions_dict[p_match[idx]] for idx in top_indices]
    top_possessions = tail_dataframes_to_sequence_length(top_possessions, sequence_length)

    top_possession_titles = []
    for i, idx in enumerate(top_indices):
        possession = top_possessions[i]
        attention_weight = attention_weights[idx] if attention_weights is not None else None
        top_possession_titles.append(_title_with_value(possession, predicted_values[idx], attention_weight))

    fig = plot_multiple_possessions(top_possessions, titles=top_possession_titles,
                                    main_title=f"Top {top_n} Actions by Predicted Value - {game_summary(selected_match)}")
    plt.tight_layout()


def visualize_top_possessions_matches(game_possessions_list: List[Tuple[Series, dict]], p_match: np.ndarray,
                                      match_ids: np.ndarray, predicted_values: np.ndarray,
                                      attention_weights: Optional[np.ndarray], top_n: int, sequence_length: int):
    top_indices = get_top_indices(predicted_values, top_n)
    top_possessions = []
    titles = []

    for idx in top_indices:
        possession_id = p_match[idx]
        match_id = int(match_ids[idx])

        match, possessions = [item for item in game_possessions_list if item[0]['game_id'] == match_id][0]
        top_possessions.append(possessions[possession_id])

        title = game_summary(match) + "\n" + _title_with_value(possessions[possession_id], predicted_values[idx],
                                                               attention_weights[idx] if attention_weights is not None else None)
        titles.append(title)

    top_possessions = tail_dataframes_to_sequence_length(top_possessions, sequence_length)
    plot_multiple_possessions(top_possessions, titles=titles, main_title=f"Top {top_n} Actions by Predicted Value - Validation Set")
    plt.tight_layout()
