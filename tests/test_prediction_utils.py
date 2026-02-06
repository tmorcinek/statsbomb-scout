"""Test script for prediction utilities."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.prediction_utils import (
    get_top_indices,
    create_top_sequences,
    normalize_predictions,
    _build_possession_row,
    find_match_by_id,
    visualize_top_sequences,
    visualize_top_sequences_matches
)
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
def processed_match(sample_game, preprocessor):
    """Process a match and return X, y, sequences."""
    match, events = sample_game
    match_id = match.get('game_id', match.name)
    X, y, sequences = preprocessor.process_match(match_id, match, events, mode=PreprocessingMode.VALIDATION)
    return match, X, y, sequences


@pytest.fixture
def sample_sequences(processed_match):
    """Return sequences from processed match."""
    _, _, _, sequences = processed_match
    return sequences


@pytest.fixture
def sample_predicted_values(processed_match):
    """Return mock predicted values matching the number of sequences."""
    _, _, y, sequences = processed_match
    # Use actual labels as predicted values for testing
    return y


@pytest.fixture
def sample_attention_weights(processed_match):
    """Return mock attention weights."""
    _, _, _, sequences = processed_match
    n_sequences = len(sequences)
    # Generate random attention weights that sum to 1 for each sequence
    weights = np.random.dirichlet(np.ones(SEQUENCE_LENGTH), size=n_sequences)
    return weights


class TestGetTopIndices:
    """Test suite for get_top_indices function."""

    def test_get_top_indices_basic(self):
        """Test basic functionality of get_top_indices."""
        values = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        top_3 = get_top_indices(values, 3)

        assert len(top_3) == 3, "Should return exactly 3 indices"
        assert top_3[0] == 3, "First index should be 3 (highest value 0.9)"
        assert top_3[1] == 1, "Second index should be 1 (second highest 0.5)"
        assert top_3[2] == 2, "Third index should be 2 (third highest 0.3)"

    def test_get_top_indices_all(self):
        """Test getting all indices without n parameter."""
        values = np.array([0.1, 0.5, 0.3])
        all_indices = get_top_indices(values)

        assert len(all_indices) == 3, "Should return all indices"
        assert all_indices[0] == 1, "First should be highest"

    def test_get_top_indices_empty(self):
        """Test with n=0."""
        values = np.array([0.1, 0.5, 0.3])
        top_0 = get_top_indices(values, 0)

        assert len(top_0) == 0, "Should return empty array"

    def test_get_top_indices_more_than_available(self):
        """Test requesting more indices than available."""
        values = np.array([0.1, 0.5])
        top_5 = get_top_indices(values, 5)

        assert len(top_5) == 2, "Should return only available indices"


class TestNormalizePredictions:
    """Test suite for normalize_predictions function."""

    def test_normalize_predictions_array(self):
        """Test normalization with simple numpy array."""
        predictions = np.array([[0.1], [0.5], [0.3]])
        values, weights = normalize_predictions(predictions)

        assert values.shape == (3,), "Should flatten to 1D"
        assert weights is None, "Should return None for weights"
        assert np.array_equal(values, np.array([0.1, 0.5, 0.3]))

    def test_normalize_predictions_dict(self):
        """Test normalization with dictionary containing attention weights."""
        predictions = {
            'value': np.array([[0.1], [0.5], [0.3]]),
            'attention_weights': np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        }
        values, weights = normalize_predictions(predictions)

        assert values.shape == (3,), "Should flatten values to 1D"
        assert weights is not None, "Should return attention weights"
        assert weights.shape == (2, 3), "Weights shape should be preserved"


class TestBuildPossessionRow:
    """Test suite for _build_possession_row function."""

    def test_build_possession_row_basic(self, sample_sequences, sample_predicted_values):
        """Test building a possession row with basic data."""
        sequence = sample_sequences[0]
        predicted_values = sample_predicted_values

        row = _build_possession_row(
            possession=sequence,
            idx=0,
            possession_id=2,
            game_id=3942819,
            predicted_values=predicted_values,
            attention_weights=None
        )

        assert isinstance(row, dict), "Should return a dictionary"
        assert 'possession_id' in row
        assert 'value' in row
        assert 'game_id' in row
        assert 'team_id' in row
        assert 'team_name' in row
        assert 'period' in row
        assert 'time_start' in row
        assert 'time_end' in row
        assert 'outcome' in row
        assert 'attention_weights' in row

        assert row['possession_id'] == 2
        assert row['game_id'] == 3942819
        assert row['attention_weights'] is None

    def test_build_possession_row_with_attention(self, sample_sequences, sample_predicted_values, sample_attention_weights):
        """Test building a possession row with attention weights."""
        sequence = sample_sequences[0]
        predicted_values = sample_predicted_values
        attention_weights = sample_attention_weights

        row = _build_possession_row(
            possession=sequence,
            idx=0,
            possession_id=2,
            game_id=3942819,
            predicted_values=predicted_values,
            attention_weights=attention_weights
        )

        assert row['attention_weights'] is not None
        assert isinstance(row['attention_weights'], str)
        # Should be a string representation of a list
        assert row['attention_weights'].startswith('[')
        assert row['attention_weights'].endswith(']')

    def test_build_possession_row_values(self, sample_sequences, sample_predicted_values):
        """Test that row values are correctly extracted."""
        sequence = sample_sequences[0]
        predicted_values = sample_predicted_values

        row = _build_possession_row(
            possession=sequence,
            idx=0,
            possession_id=2,
            game_id=3942819,
            predicted_values=predicted_values,
            attention_weights=None
        )

        # Check that team_id is an integer
        assert isinstance(row['team_id'], int)

        # Check that period is an integer
        assert isinstance(row['period'], int)
        assert row['period'] in [1, 2, 3, 4, 5]

        # Check that time values are integers
        assert isinstance(row['time_start'], int)
        assert isinstance(row['time_end'], int)
        assert row['time_end'] >= row['time_start']

        # Check outcome is a valid string
        assert isinstance(row['outcome'], str)
        assert row['outcome'] in ['Goal', 'Shot', 'Foul', 'Pass']


class TestCreateTopSequences:
    """Test suite for create_top_sequences function."""

    def test_create_top_sequences_basic(self, processed_match, sample_predicted_values):
        """Test basic functionality of create_top_sequences."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5
        )

        assert isinstance(top_df, pd.DataFrame), "Should return a DataFrame"
        assert len(top_df) == 5, "Should return exactly 5 rows"

        # Check all required columns exist
        required_columns = ['possession_id', 'value', 'attention_weights', 'game_id',
                            'team_id', 'team_name', 'period', 'time_start', 'time_end', 'outcome']
        for col in required_columns:
            assert col in top_df.columns, f"Column {col} should exist"

    def test_create_top_sequences_with_attention(self, processed_match, sample_predicted_values, sample_attention_weights):
        """Test create_top_sequences with attention weights."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=sample_attention_weights,
            head=3
        )

        assert len(top_df) == 3, "Should return exactly 3 rows"
        assert top_df['attention_weights'].notna().all(), "All attention weights should be present"

        # Check that attention weights are strings
        for weight_str in top_df['attention_weights']:
            assert isinstance(weight_str, str)
            assert weight_str.startswith('[')

    def test_create_top_sequences_ordering(self, processed_match, sample_predicted_values):
        """Test that sequences are correctly ordered by predicted value."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=10
        )

        # Check that values are in descending order
        values = top_df['value'].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1)), \
            "Values should be in descending order"

    def test_create_top_sequences_top_n_larger_than_available(self, processed_match, sample_predicted_values):
        """Test when top_n is larger than available sequences."""
        match, _, _, sequences = processed_match
        n_sequences = len(sequences)

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=n_sequences + 100  # Request more than available
        )

        assert len(top_df) == n_sequences, f"Should return all {n_sequences} sequences"

    def test_create_top_sequences_single(self, processed_match, sample_predicted_values):
        """Test getting just the top sequence."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=1
        )

        assert len(top_df) == 1, "Should return exactly 1 row"
        # Should have the highest predicted value
        max_value = sample_predicted_values.max()
        assert top_df['value'].values[0] == max_value

    def test_create_top_sequences_game_id(self, processed_match, sample_predicted_values):
        """Test that game_id is correctly set."""
        match, _, _, sequences = processed_match
        expected_game_id = match.get('game_id', match.get('match_id', 'N/A'))

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5
        )

        assert (top_df['game_id'] == expected_game_id).all(), \
            "All rows should have the correct game_id"

    def test_create_top_sequences_data_types(self, processed_match, sample_predicted_values):
        """Test that data types are correct in the output DataFrame."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5
        )

        # Check data types
        assert pd.api.types.is_integer_dtype(top_df['possession_id'])
        assert pd.api.types.is_numeric_dtype(top_df['value'])
        assert pd.api.types.is_integer_dtype(top_df['game_id'])
        assert pd.api.types.is_integer_dtype(top_df['team_id'])
        assert pd.api.types.is_string_dtype(top_df['team_name']) or pd.api.types.is_object_dtype(top_df['team_name'])
        assert pd.api.types.is_integer_dtype(top_df['period'])
        assert pd.api.types.is_integer_dtype(top_df['time_start'])
        assert pd.api.types.is_integer_dtype(top_df['time_end'])

    def test_create_top_sequences_possession_ids(self, processed_match, sample_predicted_values):
        """Test that possession IDs are extracted correctly from sequences."""
        match, _, _, sequences = processed_match

        top_df = create_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5
        )

        # Check that possession_ids are valid integers
        for pid in top_df['possession_id']:
            assert isinstance(pid, (int, np.integer))
            assert pid >= 0


class TestFindMatchById:
    """Test suite for find_match_by_id function."""

    def test_find_match_by_id_exists(self, sample_game):
        """Test finding a match that exists."""
        match, events = sample_game
        all_matches = [(match, events)]

        found_match, found_events = find_match_by_id(all_matches, 3942819)

        assert found_match.get('game_id') == 3942819
        assert len(found_events) == len(events)

    def test_find_match_by_id_not_exists(self, sample_game):
        """Test that ValueError is raised when match not found."""
        match, events = sample_game
        all_matches = [(match, events)]

        with pytest.raises(ValueError, match="Match with game_id .* not found"):
            find_match_by_id(all_matches, 999999)


class TestVisualizeTopSequences:
    """Test suite for visualize_top_sequences function."""

    def test_visualize_top_sequences_basic(self, processed_match, sample_predicted_values):
        """Test basic visualization of top sequences."""
        _, _, _, sequences = processed_match

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5
        )

        assert fig is not None, "Should return a figure"
        # Check that figure has axes
        assert len(fig.axes) > 0, "Figure should have axes"

    def test_visualize_top_sequences_with_attention(self, processed_match, sample_predicted_values, sample_attention_weights):
        """Test visualization with attention weights."""
        _, _, _, sequences = processed_match

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=sample_attention_weights,
            head=3
        )

        assert fig is not None, "Should return a figure"

    def test_visualize_top_sequences_custom_title(self, processed_match, sample_predicted_values):
        """Test visualization with custom title."""
        _, _, _, sequences = processed_match
        custom_title = "My Custom Title"

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=5,
            main_title=custom_title
        )

        assert fig is not None, "Should return a figure"
        # Check that custom title is used
        if fig._suptitle:
            assert custom_title in fig._suptitle.get_text()

    def test_visualize_top_sequences_single(self, processed_match, sample_predicted_values):
        """Test visualizing single top sequence."""
        _, _, _, sequences = processed_match

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=1
        )

        assert fig is not None, "Should return a figure"

    def test_visualize_top_sequences_many(self, processed_match, sample_predicted_values):
        """Test visualizing many sequences."""
        _, _, _, sequences = processed_match

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            head=10
        )

        assert fig is not None, "Should return a figure"

    def test_visualize_top_sequences_all(self, processed_match, sample_predicted_values):
        """Test visualizing all sequences when head=None."""
        _, _, _, sequences = processed_match

        fig = visualize_top_sequences(
            sequences=sequences,
            predicted_values=sample_predicted_values,
            attention_weights=None,
        )

        assert fig is not None, "Should return a figure"


class TestVisualizeTopSequencesMatches:
    """Test suite for visualize_top_sequences_matches function."""

    def test_visualize_top_sequences_matches_basic(self, processed_match, sample_predicted_values, sample_game):
        """Test basic visualization with match_ids."""
        match, _, _, sequences = processed_match
        match_ids = np.full(len(sequences), match.get('game_id'), dtype=np.int64)
        all_matches = [sample_game]

        fig = visualize_top_sequences_matches(
            sequences=sequences,
            match_ids=match_ids,
            predicted_values=sample_predicted_values,
            attention_weights=None,
            all_matches=all_matches,
            sequence_length=SEQUENCE_LENGTH,
            head=5
        )

        assert fig is not None, "Should return a figure"
        assert len(fig.axes) > 0, "Figure should have axes"

    def test_visualize_top_sequences_matches_with_attention(self, processed_match, sample_predicted_values, sample_attention_weights, sample_game):
        """Test visualization with attention weights."""
        match, _, _, sequences = processed_match
        match_ids = np.full(len(sequences), match.get('game_id'), dtype=np.int64)
        all_matches = [sample_game]

        fig = visualize_top_sequences_matches(
            sequences=sequences,
            match_ids=match_ids,
            predicted_values=sample_predicted_values,
            attention_weights=sample_attention_weights,
            all_matches=all_matches,
            sequence_length=SEQUENCE_LENGTH,
            head=3
        )

        assert fig is not None, "Should return a figure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
