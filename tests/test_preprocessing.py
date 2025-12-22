"""Test script to verify preprocessing flow."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import socceraction.spadl as spadl

from src.analysis.visualization import plot_possession_actions
from src.data.data_loader import load_statsbomb_socceraction_data
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.xthreat import get_default_xt_model

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@pytest.fixture(scope="module")
def sample_game():
    return next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))


@pytest.fixture(scope="module")
def preprocessor():
    return SequencePreprocessor(sequence_length=6, xt_model=get_default_xt_model())


@pytest.fixture(scope="module")
def sample_extracted_possession(sample_game, preprocessor):
    match, events = sample_game
    return preprocessor._extract_possessions(events)[0]


@pytest.fixture(scope="module")
def sample_extracted_features(preprocessor, sample_extracted_possession):
    return preprocessor._extract_features(sample_extracted_possession)


@pytest.fixture(scope="module")
def sample_normalized_features(preprocessor, sample_extracted_features):
    return preprocessor._normalize_features(sample_extracted_features)


def test_extract_possessions(sample_game, preprocessor):
    match, events = sample_game

    assert len(events) == 3485, "Number of events does not match!"

    match_id = match.get('game_id', match.name)  # Try game_id first, fallback to index
    assert match_id == 3942819, "Match ID does not match!"

    home_team_id = match.get('home_team_id')
    assert home_team_id == 941, "Home team ID does not match!"

    # Test _extract_possessions
    possessions_df = preprocessor._extract_possessions(events)
    assert len(possessions_df) == 95, "Number of possessions_df does not match!"

    # Test _extract_features
    first_possession_df = possessions_df[0]
    assert len(first_possession_df) == 13, "Number of actions in first possession does not match!"
    assert first_possession_df.iloc[0]['possession'] == 2, "First possession ID does not match!"
    assert len(first_possession_df.columns) == 28, "Number of columns in actions does not match!"

    expected_columns = ['game_id', 'event_id', 'period_id', 'team_id', 'player_id', 'type_id', 'type_name', 'index', 'timestamp', 'minute', 'second',
                        'possession', 'possession_team_id', 'possession_team_name', 'play_pattern_id', 'play_pattern_name', 'team_name', 'duration', 'extra',
                        'related_events', 'player_name', 'position_id', 'position_name', 'location', 'under_pressure', 'counterpress', 'visible_area_360',
                        'freeze_frame_360']
    assert list(first_possession_df.columns) == expected_columns, "Column names do not match!"


def test_extract_features(preprocessor, sample_extracted_possession):
    assert sample_extracted_possession.iloc[0]['possession'] == 2

    features_df = preprocessor._extract_features(sample_extracted_possession)
    print(f"Features: \n{features_df}")

    assert len(features_df) == 7, "Number of actions does not match!"
    assert len(features_df.columns) == 24, "Number of features does not match!"

    expected_columns = ['game_id',
                        'original_event_id',
                        'period_id',
                        'time_seconds',
                        'team_id',
                        'player_id',
                        'start_x',
                        'start_y',
                        'end_x',
                        'end_y',
                        'type_id',
                        'result_id',
                        'bodypart_id',
                        'action_id',
                        'dx',
                        'dy',
                        'distance',
                        'angle',
                        'time_diff',
                        'duration',
                        'under_pressure',
                        'counterpress',
                        'xG',
                        'xT']
    assert list(features_df.columns) == expected_columns, "Column names do not match!"


def test_extracted_features(sample_game, sample_extracted_features):
    actions = (spadl.add_names(sample_extracted_features))
    actions['team_name'] = "Netherlands" if sample_extracted_features.iloc[0]['team_id'] == 941 else "England"
    actions['possession'] = 2

    assert len(sample_extracted_features) == 7, "Number of actions does not match!"

    fig = plot_possession_actions(actions)
    fig.savefig('data/test/possession_2.png', dpi=300, bbox_inches='tight')
    # plt.show()


def test_normalize_features(preprocessor, sample_extracted_features):
    normalized_features = preprocessor._normalize_features(sample_extracted_features)
    assert normalized_features.shape == (7, 45), "Normalized features shape does not match!"

    expected_normalized_features_df = pd.read_csv('data/test/normalized_features_df.csv')
    assert np.allclose(expected_normalized_features_df.values, normalized_features), "Normalized features do not match expected values!"


def test_create_simple_sequence(preprocessor, sample_normalized_features):
    assert sample_normalized_features.shape == (7, 45), "Normalized features shape does not match!"

    sequence = preprocessor._create_simple_sequence(sample_normalized_features)
    assert sequence.shape == (1, 6, 45), "Normalized features shape does not match!"

    assert np.allclose(sample_normalized_features[-6:], sequence[0]), "Normalized features do not match expected values!"


def test_flow(sample_extracted_features):
    return
    if possessions_df:
        print(f"Features DataFrame shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        print(f"First row features:\n{features_df.iloc[0][['start_x', 'start_y', 'distance', 'angle', 'time_diff']]}")

    # Test _normalize_features
    print("\n--- Testing _normalize_features ---")
    if possessions_df:
        normalized = preprocessor._normalize_features(features_df)
        print(f"Normalized features shape: {normalized.shape}")
        print(f"Feature range: min={normalized.min():.3f}, max={normalized.max():.3f}")
        print(f"First action features (first 10): {normalized[0][:10]}")

    # Test _create_sequences (sliding window)
    print("\n--- Testing _create_sequences ---")
    sequences = None
    if possessions_df:
        try:
            sequences = preprocessor._create_sequences(normalized)
            print(f"Sequences shape: {sequences.shape}")
            print(f"Expected: ({len(normalized) - preprocessor.sequence_length + 1}, {preprocessor.sequence_length}, {normalized.shape[1]})")

            if len(sequences) > 0:
                print(f"First sequence shape: {sequences[0].shape}")
                print(f"First sequence first action (first 5 features): {sequences[0][0][:5]}")
        except ValueError as e:
            # Possession too short to create sliding windows — that's acceptable for some possessions_df
            print(f"_create_sequences raised: {e}")

    # Test _create_simple_sequence and _create_label (consistent API usage)
    print("\n--- Testing _create_simple_sequence and _create_label ---")
    if possessions_df:
        try:
            simple_seq = preprocessor._create_simple_sequence(normalized)
            print(f"Simple sequence shape: {simple_seq.shape}")
            label = preprocessor._create_label(features_df.tail(preprocessor.sequence_length))
            print(f"Label: {label}")
        except ValueError as e:
            print(f"Simple sequence/label creation raised: {e}")

    # Test full pipeline
    print("\n--- Testing process_match ---")
    X, y = preprocessor.process_match(match_id, home_team_id, events)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

    # Validate shapes
    print("\n--- Validation ---")
    assert X.shape[0] == y.shape[0], "Number of sequences and labels must match!"
    assert X.shape[1] == preprocessor.sequence_length, f"Sequence length must be {preprocessor.sequence_length}!"
    assert X.shape[2] > 0, "Feature dimension must be > 0!"
    print("✅ All validations passed!")

    # Do not return values from tests (pytest warns if a test returns a value)
    # The test assertions above are sufficient.

# if __name__ == "__main__":
# X, y = test_preprocessing_flow()
# print(f"\n🎉 Preprocessing flow works correctly!")
# print(f"Generated {len(X)} sequences with {X.shape[2]} features each")
