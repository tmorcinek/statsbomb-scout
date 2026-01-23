"""Test script to verify preprocessing flow."""
import numpy as np
import pandas as pd
import pytest
import socceraction.spadl as spadl

from src.analysis.visualization import plot_possession_actions
from src.data.data_loader import load_statsbomb_socceraction_data
from src.ml.preprocessing.possessions_extraction import extract_possessions
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.xthreat import get_default_xt_model

SEQUENCE_LENGTH = 6

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@pytest.fixture(scope="module")
def sample_game():
    return next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))


@pytest.fixture(scope="module")
def preprocessor():
    return SequencePreprocessor(sequence_length=SEQUENCE_LENGTH, xt_model=get_default_xt_model())


@pytest.fixture(scope="module")
def actions(sample_game, preprocessor):
    return preprocessor._extract_actions(*sample_game)


@pytest.fixture(scope="module")
def sample_extracted_action(actions):
    return actions[2]


@pytest.fixture(scope="module")
def sample_updated_action(preprocessor, sample_extracted_action):
    return preprocessor._update_action(sample_extracted_action)


@pytest.fixture(scope="module")
def shot_action(preprocessor, actions):
    return preprocessor._update_action(actions[17])


@pytest.fixture(scope="module")
def sample_normalized_features(preprocessor, sample_updated_action):
    return preprocessor._normalize_features(sample_updated_action)


def test_extract_possessions(sample_game, preprocessor):
    match, events = sample_game

    assert len(events) == 3485, "Number of events does not match!"

    match_id = match.get('game_id', match.name)  # Try game_id first, fallback to index
    assert match_id == 3942819, "Match ID does not match!"

    home_team_id = match.get('home_team_id')
    assert home_team_id == 941, "Home team ID does not match!"

    # Test _extract_possessions
    possessions_df = preprocessor._extract_actions(match, events)
    assert len(possessions_df) == 82, "Number of possessions_df does not match!"

    # Test _extract_features
    first_possession_df = possessions_df[2]
    assert len(first_possession_df) == 7, "Number of actions in first possession does not match!"
    assert first_possession_df.iloc[0]['possession'] == 2, "First possession ID does not match!"
    assert len(first_possession_df.columns) == 24, "Number of columns in actions does not match!"

    print(first_possession_df.columns)

    expected_columns = ['game_id', 'original_event_id', 'period_id', 'time_seconds', 'team_id', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_id',
                        'result_id', 'bodypart_id', 'action_id', 'team_name', 'player_name', 'possession', 'type_name', 'duration', 'under_pressure',
                        'counterpress', 'xG', 'possession_team_id', 'possession_team_name']
    assert list(first_possession_df.columns) == expected_columns, "Column names do not match!"


def test_extract_features(preprocessor, sample_extracted_action):
    assert sample_extracted_action.iloc[0]['possession'] == 2

    features_df = preprocessor._update_action(sample_extracted_action)
    print(f"Features: \n{features_df}")

    assert len(features_df) == 7, "Number of actions does not match!"
    assert len(features_df.columns) == 31, "Number of features does not match!"

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
                        'team_name',
                        'player_name',
                        'possession',
                        'type_name',
                        'duration',
                        'under_pressure',
                        'counterpress',
                        'xG',
                        'possession_team_id',
                        'possession_team_name',
                        'xT',
                        'dx',
                        'dy',
                        'distance',
                        'angle',
                        'time_diff',
                        'opposite_action']
    assert list(features_df.columns) == expected_columns, "Column names do not match!"


def test_extracted_features(sample_game, sample_updated_action):
    actions = (spadl.add_names(sample_updated_action))
    actions['team_name'] = "Netherlands" if sample_updated_action.iloc[0]['team_id'] == 941 else "England"
    actions['possession'] = 2

    assert len(sample_updated_action) == 7, "Number of actions does not match!"

    fig = plot_possession_actions(actions)
    fig.savefig('data/test/possession_2.png', dpi=300, bbox_inches='tight')
    # plt.show()


def test_normalize_features(preprocessor, sample_updated_action):
    normalized_features = preprocessor._normalize_features(sample_updated_action)
    assert normalized_features.shape == (7, 46), "Normalized features shape does not match!"

    expected_normalized_features_df = pd.read_csv('data/test/normalized_features_df.csv')
    assert np.allclose(expected_normalized_features_df.values, normalized_features), "Normalized features do not match expected values!"


def test_normalize_features_goal(preprocessor, actions):
    features = preprocessor._update_action(actions[11])
    normalized_features = preprocessor._normalize_features(features)
    assert normalized_features.shape == (20, 46), "Normalized features shape does not match!"

    expected_normalized_features_df = pd.read_csv('data/test/normalized_features_8.csv')
    assert np.allclose(expected_normalized_features_df.values, normalized_features), "Normalized features do not match expected values!"


def test_create_simple_sequence(preprocessor, sample_normalized_features):
    assert sample_normalized_features.shape == (7, 46), "Normalized features shape does not match!"

    sequence = preprocessor._create_simple_sequence(sample_normalized_features)
    assert sequence.shape == (1, 6, 46), "Normalized features shape does not match!"

    assert np.allclose(sample_normalized_features[-6:], sequence[0]), "Normalized features do not match expected values!"


def test_create_label(preprocessor, sample_updated_action):
    features = sample_updated_action.tail(SEQUENCE_LENGTH)
    label = preprocessor._create_label(features)

    xg_sum = features['xG'].sum()
    assert xg_sum == 0.0, "xG sum does not match!"

    xt_sum = features['xT'].sum()
    assert xt_sum == 0.00789534, "xT sum does not match!"

    assert label == xt_sum, "Label value does not match!"


def test_create_label_from_shot(preprocessor, actions):
    shot_possession = actions[17]
    shot_features = preprocessor._update_action(shot_possession)

    # print(shot_features)

    actions = (spadl.add_names(shot_features))
    actions["team_name"] = np.where(actions["team_id"] == 941, "Netherlands", "England")
    actions['possession'] = shot_possession['possession'].iloc[0]
    print(f"Shot Possession Actions: \n{actions}")
    fig = plot_possession_actions(actions)
    fig.savefig('data/test/possession_17.png', dpi=300, bbox_inches='tight')
    # plt.show()

    features = shot_features.tail(SEQUENCE_LENGTH)
    label = preprocessor._create_label(features)

    xg_sum = features['xG'].sum()
    assert xg_sum == 0.028932061, "xG sum does not match!"

    assert xg_sum == shot_features.iloc[-2]['xG'], "xG sum does not match!"

    assert label == xg_sum, "Label value does not match!"


def test_create_label_from_goal(preprocessor, actions):
    goal_possession = actions[11]
    shot_features = preprocessor._update_action(goal_possession)

    features = shot_features.tail(SEQUENCE_LENGTH)
    label = preprocessor._create_label(features)

    xg_sum = features['xG'].sum()
    assert xg_sum == 0.04893475, "xG sum does not match!"

    xt_sum = features['xT'].sum()
    assert xt_sum == 0.004536809999999999, "xT sum does not match!"

    assert label == 0.04893475, "Label value does not match!"


def test_process_match(sample_game, preprocessor):
    match, events = sample_game
    match_id = match.get('game_id', match.name)

    X, y, p = preprocessor.process_match(match_id, match, events)

    expected_normalized_features_df = pd.read_csv('data/test/normalized_features_df.csv').tail(SEQUENCE_LENGTH)
    assert np.allclose(expected_normalized_features_df.values, X[0]), "Normalized features do not match expected values!"

    test_y = pd.read_csv('data/test/test_y.csv')
    assert np.allclose(y, test_y['values'].values), "Y values do not match!"

    assert y[0] == 0.00789534, "Y[0] value does not match!"
    assert y[8] == 0.04893475, "Y[8] value does not match!"
    assert y[12] == 0.028932061, "Y shape does not match!"

    assert X.shape == (82, 6, 46), "X shape does not match!"
    assert y.shape == (82,), "y shape does not match!"
    assert p.shape == (82,), "p shape does not match!"

    keys = [pid for pid, df in extract_possessions(match, events).items() if len(df) >= SEQUENCE_LENGTH]
    expected_p = np.array(keys, dtype=np.int32)
    assert np.array_equal(p, expected_p), "P values (possession IDs) do not match!"


def test_process_matches(sample_game, preprocessor):
    matches = [sample_game]
    X, y, p, m = preprocessor.process_matches(matches)
    assert X.shape == (82, 6, 46), "X shape does not match!"
    assert y.shape == (82,), "y shape does not match!"
    assert p.shape == (82,), "p shape does not match!"
    assert m.shape == (82,), "m shape does not match!"

    assert np.all(m == 3942819), "Match IDs in m do not match!"
