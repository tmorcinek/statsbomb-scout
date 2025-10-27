"""Module for preprocessing event data into sequences."""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


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

    def extract_possessions(self, events_df: pd.DataFrame) -> List[pd.DataFrame]:
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

    def create_features(self, possession_df: pd.DataFrame) -> np.ndarray:
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

    def create_sequences(self, features: np.ndarray) -> List[np.ndarray]:
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

    def create_labels(self, possession_df: pd.DataFrame,
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

    def prepare_dataset(self, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline: possessions → sequences → features & labels.

        Args:
            events_df: Raw events DataFrame

        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        possessions = self.extract_possessions(events_df)

        all_sequences = []
        all_labels = []

        for possession in possessions:
            features = self.create_features(possession)
            sequences = self.create_sequences(features)

            # TODO: Generate labels for each sequence
            labels = self.create_labels(possession,
                                       list(range(len(sequences))))

            all_sequences.extend(sequences)
            all_labels.extend(labels)

        X = np.array(all_sequences)
        y = np.array(all_labels)

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.1, val_size: float = 0.2
                   ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Feature array
            y: Label array
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

