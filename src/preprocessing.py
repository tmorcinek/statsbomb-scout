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

    def _create_action_features(self, possession: pd.DataFrame, home_team_id: int) -> np.ndarray:
        """
        Create ML-ready features from possession events.

        Args:
            possession: DataFrame with possession events
            home_team_id: ID of home team

        Returns:
            Feature array of shape (n_actions, n_features)
        """
        # Convert possession to SPADL actions
        actions = spadl.statsbomb.convert_to_actions(possession.copy(), home_team_id)

        if len(actions) == 0:
            return np.array([])

        features = []

        for idx, action in actions.iterrows():
            action_features = []

            # 1. Spatial features (normalized to 0-1)
            action_features.extend([
                action['start_x'] / spadl_config.field_length,
                action['start_y'] / spadl_config.field_width,
                action['end_x'] / spadl_config.field_length,
                action['end_y'] / spadl_config.field_width
            ])

            # 2. Geometric features
            dx = action['end_x'] - action['start_x']
            dy = action['end_y'] - action['start_y']
            distance = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.arctan2(dy, dx)

            action_features.extend([
                distance / spadl_config.field_length,
                np.sin(angle),
                np.cos(angle)
            ])

            # 3. Temporal features
            if idx > 0:
                prev_action = actions.iloc[idx - 1]
                time_diff = action['time_seconds'] - prev_action['time_seconds']
                action_features.append(min(time_diff / 10.0, 1.0))
            else:
                action_features.append(0.0)

            # 4. Contextual features from original possession events
            possession_event = possession[possession['timestamp'] == action.get('timestamp')]
            under_pressure = False
            counterpress = False

            if not possession_event.empty:
                under_pressure = possession_event.iloc[0].get('under_pressure', False)
                counterpress = possession_event.iloc[0].get('counterpress', False)

            action_features.extend([
                float(under_pressure),
                float(counterpress)
            ])

            # 5. Categorical features (one-hot encoding)
            type_id = action['type_id']
            type_onehot = [1.0 if i == type_id else 0.0 for i in range(20)]
            action_features.extend(type_onehot)

            result_id = action['result_id']
            result_onehot = [1.0 if i == result_id else 0.0 for i in range(3)]
            action_features.extend(result_onehot)

            bodypart_id = action['bodypart_id']
            bodypart_onehot = [1.0 if i == bodypart_id else 0.0 for i in range(5)]
            action_features.extend(bodypart_onehot)

            features.append(action_features)

        return np.array(features, dtype=np.float32)

    def _create_features(self, possession_df: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors for each action in possession.

        Args:
            possession_df: DataFrame with events from one possession

        Returns:
            Feature array of shape (n_actions, n_features)
        """
        features = []
        previous_timestamp = None

        for idx, row in possession_df.iterrows():
            action_features = []

            # 1. Location coordinates (x, y)
            location = row.get('location', [0, 0])
            x, y = location[0] if len(location) > 0 else 0, location[1] if len(location) > 1 else 0
            action_features.extend([x, y])

            # 2. End location (for passes/carries)
            end_location = row.get('pass_end_location') or row.get('carry_end_location', [x, y])
            x_end = end_location[0] if len(end_location) > 0 else x
            y_end = end_location[1] if len(end_location) > 1 else y
            action_features.extend([x_end, y_end])

            # 3. Action type (one-hot or integer encoding)
            action_type = row['type']
            if action_type not in self.action_type_mapping:
                self.action_type_mapping[action_type] = len(self.action_type_mapping)
            action_features.append(self.action_type_mapping[action_type])

            # 4. Pass/movement metrics
            distance = np.sqrt((x_end - x) ** 2 + (y_end - y) ** 2)
            angle = np.arctan2(y_end - y, x_end - x)
            action_features.extend([distance, angle])

            # 5. Time delta from previous action
            current_timestamp = pd.to_datetime(row['timestamp'])
            if previous_timestamp is not None:
                time_delta = (current_timestamp - previous_timestamp).total_seconds()
            else:
                time_delta = 0.0
            action_features.append(time_delta)
            previous_timestamp = current_timestamp

            # 6. Contextual flags
            under_pressure = 1.0 if row.get('under_pressure', False) else 0.0
            action_features.append(under_pressure)

            features.append(action_features)

        return np.array(features, dtype=np.float32)

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
