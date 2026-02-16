"""Test script to verify preprocessing flow."""
import numpy as np
import pandas as pd
import pytest
import socceraction.spadl as spadl

from src.analysis.visualization import plot_possession_actions
from src.data.data_loader import load_statsbomb_socceraction_data
from src.ml.preprocessing.sequence_preprocessor import SequencePreprocessor, PreprocessingMode
from src.ml.preprocessing.xthreat import get_default_xt_model

SEQUENCE_LENGTH = 6
MINIMUM_SEQUENCE_LENGTH = 3

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


@pytest.fixture(scope="module")
def sample_game() -> tuple[pd.Series, pd.DataFrame]:
    return next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))


@pytest.fixture(scope="module")
def preprocessor() -> SequencePreprocessor:
    return SequencePreprocessor(
        sequence_length=SEQUENCE_LENGTH,
        minimum_sequence_length=MINIMUM_SEQUENCE_LENGTH,
        xt_model=get_default_xt_model()
    )


@pytest.fixture(scope="module")
def actions(sample_game, preprocessor) -> dict[int, pd.DataFrame]:
    return preprocessor._extract_actions(*sample_game)


@pytest.fixture(scope="module")
def sample_extracted_action(actions) -> pd.DataFrame:
    return actions[2]


@pytest.fixture(scope="module")
def sample_updated_action(preprocessor, sample_extracted_action) -> pd.DataFrame:
    return preprocessor._update_action(sample_extracted_action)


@pytest.fixture(scope="module")
def shot_action(preprocessor, actions) -> pd.DataFrame:
    return preprocessor._update_action(actions[17])


@pytest.fixture(scope="module")
def sample_normalized_features(preprocessor, sample_updated_action) -> np.ndarray:
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
    assert len(possessions_df) == 97, "Number of possessions_df does not match!"

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

    sequences_ranges = preprocessor._create_evaluation_sequences(sample_normalized_features)

    assert len(sequences_ranges) == 1, "Number of sequences does not match!"

    sequence_1, _ = sequences_ranges[0]
    assert sequence_1.shape == (6, 46), "Sequences shape does not match!"

    assert np.allclose(sample_normalized_features[-6:], sequence_1), "Normalized features do not match expected values!"


def test_long_possession(preprocessor, actions, sample_game):
    long_possession = actions[109]
    assert len(long_possession) == 32, "Number of actions in penalty possession does not match!"

    match, events = sample_game
    match_id = match.get('game_id', match.name)

    possession_events = events[events['possession'] == 109]

    assert len(possession_events) == 58 , "Number of events in possession does not match!"

    X, y, p = preprocessor.process_match(match_id, match, possession_events, mode=PreprocessingMode.PLAYER_EVALUATION)

    assert p[0].shape == (6, 31), "Sequence shape does not match!"
    assert p[1].shape == (6, 31), "Sequence shape does not match!"
    assert p[2].shape == (6, 31), "Sequence shape does not match!"
    assert p[3].shape == (6, 31), "Sequence shape does not match!"
    assert p[4].shape == (6, 31), "Sequence shape does not match!"

    assert X.shape == (5, 6, 46), "X shape does not match!"
    assert y.shape == (5,), "y shape does not match!"
    assert p.shape[0] == 5, "p shape does not match!"


def test_create_label(preprocessor, sample_updated_action):
    features = sample_updated_action.tail(SEQUENCE_LENGTH)
    label = preprocessor._create_label(features)

    xg_sum = features['xG'].sum()
    assert xg_sum == 0.0, "xG sum does not match!"

    xt_sum = features['xT'].sum()
    assert xt_sum == 0.12394768, "xT sum does not match!"

    assert label == 0.02385149, "Label value does not match!"


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

    assert label == 0.25745362, "Label value does not match!"


def test_create_label_from_goal(preprocessor, actions):
    goal_possession = actions[11]
    shot_features = preprocessor._update_action(goal_possession)

    features = shot_features.tail(SEQUENCE_LENGTH)
    label = preprocessor._create_label(features)

    xg_sum = features['xG'].sum()
    assert xg_sum == 0.04893475, "xG sum does not match!"

    xt_sum = features['xT'].sum()
    assert xt_sum == 0.14785314, "xT sum does not match!"

    assert label == 0.04893475, "Label value does not match!"


def test_create_label_from_penalty(preprocessor, actions):
    goal_possession = actions[109]
    shot_features = preprocessor._update_action(goal_possession).tail(8)

    label = preprocessor._create_label(shot_features)
    shot_features = shot_features[['original_event_id', 'xG', 'xT', 'type_name', 'player_name', 'opposite_action']]

    print(f"Goal Possession Actions: \n{shot_features}")


def test_watkins_goal_xg(sample_game):
    game, events = sample_game

    watkins_goal_event = events[events['event_id'] == '42dc6a40-991d-4a08-9aa1-64c2dd8a807e'].iloc[0]

    extra = watkins_goal_event['extra']
    shot = extra['shot']
    assert shot.get('statsbomb_xg', 'N/A') == 0.035494413, "xG does not match!"
    assert shot.get('body_part', {}).get('name', 'N/A') == 'Right Foot', "Body part does not match!"
    assert shot.get('technique', {}).get('name', 'N/A') == 'Normal', "Technique does not match!"
    assert shot.get('type', {}).get('name', 'N/A') == 'Open Play', "Shot type does not match!"
    assert shot.get('outcome', {}).get('name', 'N/A') == 'Goal', "Outcome does not match!"


def test_process_match(sample_game, preprocessor):
    match, events = sample_game
    match_id = match.get('game_id', match.name)

    X, y, p = preprocessor.process_match(match_id, match, events, mode=PreprocessingMode.VALIDATION)

    assert X.shape == (97, 6, 46), "X shape does not match!"
    assert y.shape == (97,), "y shape does not match!"
    assert p.shape == (97,), "p shape does not match!"

    expected_normalized_features_df = pd.read_csv('data/test/normalized_features_df.csv').tail(SEQUENCE_LENGTH)
    assert np.allclose(expected_normalized_features_df.values, X[0]), "Normalized features do not match expected values!"

    test_y = pd.read_csv('data/test/test_y.csv')
    assert np.allclose(y, test_y['values'].values), "Y values do not match!"

    assert y[0] == 0.02385149, "Y[0] value does not match!"
    assert y[8] == 0.04893475, "Y[8] value does not match!"
    assert y[12] == 0.01248344, "Y[12] value does not match!"
    assert y[25] == 0.01870347, "Y[27] value does not match!"

    assert all(isinstance(seq, pd.DataFrame) for seq in p), "All sequences should be DataFrames"
    assert all(len(seq) <= SEQUENCE_LENGTH for seq in p), f"All sequences should have length <= {SEQUENCE_LENGTH}"
    assert all(len(seq) >= MINIMUM_SEQUENCE_LENGTH for seq in p), f"All sequences should have length >= {MINIMUM_SEQUENCE_LENGTH}"

def test_process_match_player_evaluation(sample_game, preprocessor):
    match, events = sample_game
    match_id = match.get('game_id', match.name)

    X, y, p = preprocessor.process_match(match_id, match, events, mode=PreprocessingMode.PLAYER_EVALUATION)

    assert X.shape == (362, 6, 46), "X shape does not match!"
    assert y.shape == (362,), "y shape does not match!"
    assert p.shape == (362,), "p shape does not match!"

def test_process_match_number_of_possessions(sample_game, preprocessor):
    match, events = sample_game
    match_id = match.get('game_id', match.name)

    X, y, p = preprocessor.process_match(match_id, match, events)

    assert X.shape == (1657, 6, 46), "X shape does not match!"
    assert y.shape == (1657,), "y shape does not match!"
    assert p.shape == (1657,), "p shape does not match!"


def test_process_matches(sample_game, preprocessor):
    matches = [sample_game]
    X, y, p, m = preprocessor.process_matches(matches, mode=PreprocessingMode.VALIDATION)
    assert X.shape == (97, 6, 46), "X shape does not match!"
    assert y.shape == (97,), "y shape does not match!"
    assert p.shape == (97,), "p shape does not match!"
    assert m.shape == (97,), "m shape does not match!"

    assert np.all(m == 3942819), "Match IDs in m do not match!"


def test_process_matches_training(sample_game, preprocessor):
    matches = [sample_game]
    X, y, p, m = preprocessor.process_matches(matches, mode=PreprocessingMode.TRAINING)
    assert X.shape == (1657, 6, 46), "X shape does not match!"
    assert y.shape == (1657,), "y shape does not match!"
    assert p.shape == (1657,), "p shape does not match!"
    assert m.shape == (1657,), "m shape does not match!"

    assert np.all(m == 3942819), "Match IDs in m do not match!"


if __name__ == '__main__':
    pytest.main([__file__])