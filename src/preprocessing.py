"""Module for preprocessing event data into sequences."""

from typing import List, Tuple, Generator, Union

import numpy as np
import pandas as pd
import socceraction.spadl as spadl
import socceraction.spadl.config as spadl_config


class SequencePreprocessor:
    """Preprocesses event data into fixed-length sequences with features."""

    def __init__(self, sequence_length: int = 10, minimum_possession_length: int = 2):
        self.sequence_length = sequence_length
        self.minimum_possession_length = minimum_possession_length
        self.action_type_mapping = {}

    def _extract_possessions(self, events_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Extract possession phases from single match data.

        Filters out opponent events (like Pressure, Foul Committed) that are recorded
        within possession but don't belong to the possessing team.

        Args:
            events_df: DataFrame with events

        Returns:
            List of DataFrames, each representing one possession
        """
        possessions = []

        for possession_id, possession_group in events_df.groupby("possession"):
            # Get the team that has possession
            possession_team = possession_group['possession_team_name'].iloc[0]

            # Filter events to only include actions by the possessing team
            team_events = possession_group[possession_group['team_name'] == possession_team]

            # Only include possessions with minimum number of events
            if len(team_events) >= self.minimum_possession_length:
                possessions.append(team_events.copy())

        return possessions

    def _extract_features(self, possession: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
        """
        Extract features from possession events and return as DataFrame.

        Args:
            possession: DataFrame with possession events
            home_team_id: ID of home team

        Returns:
            DataFrame with extracted features (not normalized)
        """
        # Convert possession to SPADL actions
        actions = spadl.statsbomb.convert_to_actions(possession, home_team_id)

        # 1) Geometry: dx, dy, distance, angle
        actions["dx"] = actions["end_x"] - actions["start_x"]
        actions["dy"] = actions["end_y"] - actions["start_y"]
        actions["distance"] = np.sqrt(actions["dx"] ** 2 + actions["dy"] ** 2)
        actions["angle"] = np.arctan2(actions["dy"], actions["dx"])

        # 2) Time difference between actions
        actions["time_diff"] = actions["time_seconds"].diff().fillna(0.0)

        subset = possession.filter(['event_id', 'duration', 'under_pressure', 'counterpress'])
        actions = (
            actions
            .merge(subset, left_on='original_event_id', right_on='event_id', how='left')
            .drop(columns='event_id')
            .assign(
                duration=lambda df: df['duration'].fillna(0.0),
                under_pressure=lambda df: df['under_pressure'].fillna(False),
                counterpress=lambda df: df['counterpress'].fillna(False)
            )
        )

        return actions

    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize features and convert to numpy array with one-hot encoding.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Normalized feature array of shape (n_actions, n_features)
        """
        # zero hot encoding sizes
        n_types = len(spadl_config.actiontypes)
        n_results = len(spadl_config.results)
        n_bodyparts = len(spadl_config.bodyparts)
        cap_time = 10.0

        # 1. Spatial features (normalized to 0-1)
        spatial = features_df[['start_x', 'start_y', 'end_x', 'end_y']].values / [
            spadl_config.field_length, spadl_config.field_width,
            spadl_config.field_length, spadl_config.field_width
        ]

        # 2. Geometric features
        geometric = np.column_stack([
            features_df['distance'].values / spadl_config.field_length,
            np.sin(features_df['angle'].values),
            np.cos(features_df['angle'].values)
        ])

        # 3. Temporal features (capped at 10 seconds)
        temporal = np.minimum(features_df['time_diff'].values / cap_time, 1.0).reshape(-1, 1)

        # 4. Contextual features (boolean to float)
        contextual = features_df[['under_pressure', 'counterpress']].astype(float).values

        # 5. Categorical features (one-hot encoding)
        def one_hot_numpy(ids, K, dtype=np.uint8):
            a = np.asarray(ids)

            valid = (~np.isnan(a)) if a.dtype.kind == "f" else (a >= 0)
            idx = np.where(valid, a, -1).astype(np.int64)

            out = np.zeros((len(a), K), dtype=dtype)
            rows = np.nonzero(valid)[0]
            out[rows, idx[rows]] = 1
            return out

        type_onehot = one_hot_numpy(features_df["type_id"], n_types)  # np.uint8
        result_onehot = one_hot_numpy(features_df["result_id"], n_results)
        bodypart_onehot = one_hot_numpy(features_df["bodypart_id"], n_bodyparts)

        return np.concatenate([
            spatial,
            geometric,
            temporal,
            contextual,
            type_onehot,
            result_onehot,
            bodypart_onehot
        ], axis=1).astype(np.float32)

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

    def process_match(self, match_id: int, home_team_id: int, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single match into sequences.

        Args:
            match_id: ID of the match
            home_team_id: ID of the home team
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
            # features = self._create_features(possession)
            features = spadl.statsbomb.convert_to_actions(possession, home_team_id)

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

    def process_matches(self, matches: Union[Generator[Tuple[pd.Series, pd.DataFrame], None, None], List[Tuple[pd.Series, pd.DataFrame]]]) -> Tuple[
        np.ndarray, np.ndarray]:
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

        for match, events_df in matches:
            X_match, y_match = self.process_match(match['match_id'], match['home_team_id'], events_df)

            if len(X_match) > 0:
                all_X.append(X_match)
                all_y.append(y_match)

        # Concatenate all sequences
        X = np.concatenate(all_X) if all_X else np.array([])
        y = np.concatenate(all_y) if all_y else np.array([])

        print(f"\nTotal sequences from all matches: {len(X)}")

        return X, y
