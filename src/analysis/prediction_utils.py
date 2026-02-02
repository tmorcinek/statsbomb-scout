from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.analysis.game_utils import game_summary, get_action_outcomes, get_period_offset
from src.analysis.visualization import plot_multiple_possessions, _title_with_value
from src.ml.preprocessing.possessions_extraction import extract_possessions, tail_dataframes_to_sequence_length


def find_match_by_id(all_matches: List[Tuple[pd.Series, pd.DataFrame]], match_id: int) -> Tuple[pd.Series, pd.DataFrame]:
    for match, events_df in all_matches:
        if match.get('game_id') == match_id:
            return match, events_df
    raise ValueError(f"Match with game_id {match_id} not found")


def get_top_indices(predicted_values: np.ndarray, head: Optional[int] = None) -> np.ndarray:
    sorted_indices = np.argsort(predicted_values)[::-1]
    return sorted_indices[:head] if head is not None else sorted_indices


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


def _build_possession_row(possession: DataFrame, idx: int, possession_id: int, game_id: int,
                          predicted_values: np.ndarray, attention_weights: Optional[np.ndarray]) -> dict:
    first_action = possession.iloc[0]
    last_action = possession.iloc[-1]

    offset = get_period_offset(first_action['period_id'])

    start_time = int(first_action['time_seconds']) + offset
    end_time = int(last_action['time_seconds']) + offset

    outcome = get_action_outcomes(possession)
    weights_str = str(list(attention_weights[idx])) if attention_weights is not None else None

    return {
        'possession_id': possession_id,
        'value': predicted_values[idx],
        'attention_weights': weights_str,
        'game_id': game_id,
        'team_id': int(first_action['possession_team_id']),
        'team_name': first_action.get('possession_team_name', 'Unknown'),
        'period': int(first_action['period_id']),
        'time_start': start_time,
        'time_end': end_time,
        'outcome': outcome
    }


def create_top_sequences(sequences: np.ndarray, predicted_values: np.ndarray, attention_weights: Optional[np.ndarray],
                         match_ids: Optional[np.ndarray] = None, head: Optional[int] = None) -> DataFrame:
    data = []
    for idx in get_top_indices(predicted_values, head):
        sequence = sequences[idx]
        possession_id = sequence.iloc[0]['possession']
        game_id = int(match_ids[idx]) if match_ids is not None else sequence.iloc[0]['game_id']

        row = _build_possession_row(sequence, idx, possession_id, game_id, predicted_values, attention_weights)
        data.append(row)

    return pd.DataFrame(data)


def visualize_top_sequences(sequences: np.ndarray, predicted_values: np.ndarray,
                            attention_weights: Optional[np.ndarray], head: Optional[int] = None, main_title: Optional[str] = None):
    top_indices = get_top_indices(predicted_values, head)

    top_sequences = []
    top_sequence_titles = []
    for idx in top_indices:
        sequence = sequences[idx]
        top_sequences.append(sequence)
        attention_weight = attention_weights[idx] if attention_weights is not None else None
        top_sequence_titles.append(_title_with_value(sequence, predicted_values[idx], attention_weight))

    if main_title is None:
        n_sequences = len(top_indices)
        main_title = f"Top {n_sequences} Sequences by Predicted Value"

    fig = plot_multiple_possessions(top_sequences, titles=top_sequence_titles, main_title=main_title)
    plt.tight_layout()
    return fig


def visualize_top_sequences_matches(sequences: np.ndarray, match_ids: np.ndarray,
                                    predicted_values: np.ndarray, attention_weights: Optional[np.ndarray],
                                    all_matches: List[Tuple[Series, pd.DataFrame]], sequence_length: int,
                                    head: Optional[int] = None, main_title: Optional[str] = None):
    top_indices = get_top_indices(predicted_values, head)

    top_sequences = []
    titles = []
    for idx in top_indices:
        sequence = tail_dataframes_to_sequence_length([sequences[idx]], sequence_length)[0]
        top_sequences.append(sequence)

        match, _ = find_match_by_id(all_matches, int(match_ids[idx]))
        attention_weight = attention_weights[idx] if attention_weights is not None else None
        title = game_summary(match) + "\n" + _title_with_value(sequence, predicted_values[idx], attention_weight)
        titles.append(title)

    if main_title is None:
        n_sequences = len(top_indices)
        main_title = f"Top {n_sequences} Actions by Predicted Value - Validation Set"

    fig = plot_multiple_possessions(top_sequences, titles=titles, main_title=main_title)
    plt.tight_layout()
    return fig
