"""Module for preprocessing event data into sequences."""

from typing import List, Tuple, Generator, Union

import numpy as np
import pandas as pd


class SequencePreprocessor:
    """Preprocesses event data into fixed-length sequences with features."""

    def __init__(self, sequence_length: int = 10):
        """
        Initialize preprocessor.

        Args:
            sequence_length: Number of actions in each sequence
        """
        self.sequence_length = sequence_length
        self.action_type_mapping = {}

    def _extract_possessions(self, events_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Extract possession phases from event data.

        Args:
            events_df: DataFrame with events

        Returns:
            List of DataFrames, each representing one possession
        """
        possessions = []

        # TODO: Implement possession extraction logic
        # Group events by possession_team and identify phase changes
        # Example: events_df.groupby(['match_id', 'possession'])

        return possessions

    def _create_features(self, possession_df: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors for each action in possession.

        Args:
            possession_df: DataFrame with events from one possession

        Returns:
            Feature array of shape (n_actions, n_features)
        """
        features = []

        for idx, row in possession_df.iterrows():
            action_features = []

            # TODO: Extract features for each action
            # 1. Coordinates (x_start, y_start, x_end, y_end)
            # action_features.extend([row['location'][0], row['location'][1], ...])

            # 2. Action type (encoded as integer)
            # action_type = self._encode_action_type(row['type'])
            # action_features.append(action_type)

            # 3. Pass length and angle (if applicable)
            # pass_length, pass_angle = self._calculate_pass_metrics(row)
            # action_features.extend([pass_length, pass_angle])

            # 4. Time delta from previous action
            # time_delta = self._calculate_time_delta(idx, possession_df)
            # action_features.append(time_delta)

            # 5. Under pressure flag
            # under_pressure = 1 if row.get('under_pressure', False) else 0
            # action_features.append(under_pressure)

            features.append(action_features)

        return np.array(features)

    def _create_sequences(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Create fixed-length sequences from features.

        Args:
            features: Feature array

        Returns:
            List of sequences, each of length sequence_length
        """
        sequences = []

        # TODO: Create sliding windows of length sequence_length
        for i in range(len(features) - self.sequence_length + 1):
            sequence = features[i:i + self.sequence_length]
            sequences.append(sequence)

        return sequences

    def _create_labels(self, possession_df: pd.DataFrame,
                       sequence_indices: List[int]) -> np.ndarray:
        """
        Create labels for sequences (xG of shot or 0).

        Args:
            possession_df: DataFrame with possession events
            sequence_indices: Starting indices of sequences

        Returns:
            Array of labels
        """
        labels = []

        for idx in sequence_indices:
            end_idx = idx + self.sequence_length

            # TODO: Check if sequence ends with a shot
            # If yes, use xG value; if no, use 0
            # Example: check if possession_df.iloc[end_idx-1]['type'] == 'Shot'
            # label = possession_df.iloc[end_idx-1].get('shot_statsbomb_xg', 0)

            labels.append(0)  # Placeholder

        return np.array(labels)

    def process_match(self, match_id: int, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single match into sequences.

        Args:
            match_id: ID of the match
            events_df: DataFrame with events from one match

        Returns:
            Tuple of (X, y) where:
                - X is array of sequences with shape (n_sequences, sequence_length, n_features)
                - y is array of labels with shape (n_sequences,)
        """
        print(f"Processing match {match_id}: {len(events_df)} events")

        # Extract possessions from match
        possessions = self._extract_possessions(events_df)

        all_sequences = []
        all_labels = []

        for possession in possessions:
            # Create features for each action in possession
            features = self._create_features(possession)

            # Create sequences from features
            sequences = self._create_sequences(features)

            # Generate labels for each sequence
            labels = self._create_labels(possession, list(range(len(sequences))))

            all_sequences.extend(sequences)
            all_labels.extend(labels)

        X = np.array(all_sequences) if all_sequences else np.array([]).reshape(0, self.sequence_length, 0)
        y = np.array(all_labels) if all_labels else np.array([])

        print(f"  → Generated {len(X)} sequences")

        return X, y

    def process_matches(self, matches: Union[Generator[Tuple[int, pd.DataFrame], None, None], List[Tuple[int, pd.DataFrame]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process multiple matches from generator or list into sequences.

        Args:
            matches: Generator or list yielding/containing (match_id, events_df) tuples

        Returns:
            Tuple of (X, y) where:
                - X is array of all sequences with shape (n_sequences, sequence_length, n_features)
                - y is array of all labels with shape (n_sequences,)
        """
        all_X = []
        all_y = []

        for match_id, events_df in matches:
            X_match, y_match = self.process_match(match_id, events_df)

            if len(X_match) > 0:
                all_X.append(X_match)
                all_y.append(y_match)

        # Concatenate all sequences
        X = np.concatenate(all_X) if all_X else np.array([])
        y = np.concatenate(all_y) if all_y else np.array([])

        print(f"\nTotal sequences from all matches: {len(X)}")

        return X, y
