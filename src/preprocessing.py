"""Module for preprocessing event data into sequences."""

from typing import List, Tuple, Generator, Union, Optional

import numpy as np
import pandas as pd
import socceraction.spadl as spadl
import socceraction.spadl.config as spadl_config

import warnings

from socceraction.xthreat import ExpectedThreat

warnings.filterwarnings('ignore', category=FutureWarning, module='socceraction')
pd.set_option('future.no_silent_downcasting', True)

class SequencePreprocessor:
    """Preprocesses event data into fixed-length sequences with features."""

    def __init__(
        self,
        sequence_length: int = 10,
        minimum_possession_length: int = 10,
        xt_model: ExpectedThreat = None
    ):
        """
        Initialize preprocessor.

        Args:
            sequence_length: Number of actions in each sequence
            minimum_possession_length: Minimum number of actions in a possession to process
            xt_model: Optional pre-trained xT model. If None, uses default model.
        """
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

        Converts StatsBomb events to SPADL actions and adds:
        - Geometric features (dx, dy, distance, angle)
        - Temporal features (time_diff between actions)
        - Contextual features (duration, under_pressure, counterpress)

        Args:
            possession: DataFrame with StatsBomb possession events
            home_team_id: ID of home team (required for SPADL conversion)

        Returns:
            DataFrame with SPADL actions and additional computed features (not normalized)
        """
        # Convert possession to SPADL actions
        actions = spadl.statsbomb.convert_to_actions(possession, home_team_id, xy_fidelity_version=2)

        # 1) Geometry: dx, dy, distance, angle
        actions["dx"] = actions["end_x"] - actions["start_x"]
        actions["dy"] = actions["end_y"] - actions["start_y"]
        actions["distance"] = np.sqrt(actions["dx"] ** 2 + actions["dy"] ** 2)
        actions["angle"] = np.arctan2(actions["dy"], actions["dx"])

        # 2) Time difference between actions
        actions["time_diff"] = actions["time_seconds"].diff().fillna(0.0)

        # 3) Merge contextual features
        subset = possession[['event_id', 'duration', 'under_pressure', 'counterpress']]
        actions = actions.merge(subset, left_on='original_event_id', right_on='event_id', how='left').drop(columns='event_id')
        actions = actions.fillna({'duration': 0.0, 'under_pressure': False, 'counterpress': False})

        return actions

    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize features and convert to numpy array with one-hot encoding.

        Creates ML-ready feature vectors with:
        - Spatial features (4): normalized start_x, start_y, end_x, end_y (0-1)
        - Geometric features (3): normalized distance, sin(angle), cos(angle)
        - Temporal features (1): time_diff capped at 10s and normalized
        - Contextual features (2): under_pressure, counterpress as floats
        - Categorical features (~28): one-hot encoded type_id, result_id, bodypart_id

        Args:
            features_df: DataFrame with extracted features from _extract_features()

        Returns:
            Normalized feature array of shape (n_actions, ~38 features)
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

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """
        Create fixed-length sequences using efficient sliding window with NumPy strides.

        Uses zero-copy view (np.lib.stride_tricks.sliding_window_view) for memory efficiency.
        This creates overlapping sequences without duplicating data in memory.

        Example with 7 actions and sequence_length=3:
            Input:  [A1, A2, A3, A4, A5, A6, A7]
            Output: [[A1, A2, A3],
                     [A2, A3, A4],
                     [A3, A4, A5],
                     [A4, A5, A6],
                     [A5, A6, A7]]
            → 5 sequences (7 - 3 + 1)

        Args:
            features: Feature array of shape (n_actions, n_features)
                     Must have at least sequence_length rows

        Returns:
            Array of shape (n_sequences, sequence_length, n_features) where:
                n_sequences = n_actions - sequence_length + 1

        Raises:
            ValueError: If features has fewer rows than sequence_length

        Performance:
            - O(1) time complexity (view creation, no copying)
            - O(1) space complexity (shares memory with input array)
        """
        if len(features) < self.sequence_length:
            raise ValueError(f"Insufficient number of actions ({len(features)}) for sequence length {self.sequence_length}.")

        return np.lib.stride_tricks.sliding_window_view(features, (self.sequence_length, features.shape[1])).squeeze(1)

    def _create_labels(self, actions_df: pd.DataFrame, original_events: pd.DataFrame) -> np.ndarray:
        """
        Create labels for sequences based on possession outcome.

        For each sequence, the label represents the expected value:
        - If sequence ends with a GOAL: 1.0
        - If sequence ends with a SHOT: shot_statsbomb_xg value (0.0-1.0)
        - Otherwise: delta_xT (change in Expected Threat)

        Expected Threat (xT) measures how much an action increases goal probability.
        Δ_xT = xT_end - xT_start (can be negative if action moves away from goal).

        Args:
            actions_df: DataFrame with SPADL actions from _extract_features()
                       Must contain 'type_name', 'original_event_id' columns
            original_events: DataFrame with original StatsBomb events
                            Must contain 'event_id', 'shot_statsbomb_xg', 'shot_outcome_name'

        Returns:
            Array of labels with shape (n_sequences,) where each value is:
                - 1.0 if sequence ends with a goal
                - xG (0.0-1.0) if sequence ends with a shot (non-goal)
                - Δ_xT if sequence ends with another action

        Example:
            For 5 actions with sequence_length=3:
                Actions: [pass, pass, dribble, shot(xG=0.3), goal]
                Sequences: [[pass, pass, dribble],  [pass, dribble, shot],  [dribble, shot, goal]]
                Labels:    [0.05 (Δ_xT),            0.3 (xG),                1.0 (goal)]
        """
        n_sequences = len(actions_df) - self.sequence_length + 1
        labels = np.zeros(n_sequences, dtype=np.float32)

        # 1) Create mappings for shots and goals
        xg_map = {}
        goal_events = set()

        if 'shot_statsbomb_xg' in original_events.columns:
            shot_events = original_events[original_events['shot_statsbomb_xg'].notna()].copy()
            xg_map = dict(zip(shot_events['id'], shot_events['shot_statsbomb_xg']))

            # Identify goals
            if 'shot_outcome' in original_events.columns:
                # shot_outcome is a dict with 'name' key
                goals = shot_events[shot_events['shot_outcome'].apply(
                    lambda x: isinstance(x, dict) and x.get('name') == 'Goal'
                )]
                goal_events = set(goals['id'])

        # 2) Calculate xT for all actions using the model
        delta_xt = calculate_delta_xt(actions_df, self.xt_model)

        # 3) For each sequence, assign label based on last action
        for i in range(n_sequences):
            end_idx = i + self.sequence_length - 1  # Last action in sequence
            action = actions_df.iloc[end_idx]
            original_event_id = action['original_event_id']

            # Priority 1: Check if it's a goal
            if original_event_id in goal_events:
                labels[i] = 1.0

            # Priority 2: Check if it's a shot (non-goal)
            elif action['type_name'] == 'shot':
                labels[i] = xg_map.get(original_event_id, 0.0)

            # Priority 3: Use delta xT for other actions
            else:
                labels[i] = delta_xt.iloc[end_idx]

        return labels

    def process_match(self, match_id: int, home_team_id: int, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single match into ML-ready sequences.

        Pipeline:
        1. Extract possessions (filter by team)
        2. For each possession:
           a. Extract features (_extract_features: StatsBomb → SPADL + computed features)
           b. Normalize features (_normalize_features: DataFrame → numpy array)
           c. Create sequences (_create_sequences: sliding window)
           d. Create labels (_create_labels: goal=1.0, shot=xG, other=Δ_xT)
        3. Concatenate all sequences from all possessions

        Args:
            match_id: ID of the match (for logging)
            home_team_id: ID of the home team (required for SPADL conversion)
            events_df: DataFrame with StatsBomb events from one match

        Returns:
            Tuple of (X, y) where:
                - X is array of sequences with shape (n_sequences, sequence_length, ~45)
                - y is array of labels with shape (n_sequences,) where:
                    * 1.0 = sequence ends with a goal
                    * 0.0-1.0 = sequence ends with a shot (xG value)
                    * Δ_xT = sequence ends with other action (can be negative)
        """
        print(f"Processing match {match_id}: {len(events_df)} events")

        # Extract possessions from match
        possessions = self._extract_possessions(events_df)

        all_sequences = []
        all_labels = []

        for possession in possessions:
            # Extract features
            features_df = self._extract_features(possession, home_team_id)
            if len(features_df) < self.sequence_length:
                continue

            # Normalize features
            features = self._normalize_features(features_df)

            # Create sequences from features (sliding window)
            sequences = self._create_sequences(features)

            # Generate labels for each sequence
            labels = self._create_labels(features_df, list(range(len(sequences))))

            all_sequences.append(sequences)
            all_labels.append(labels)

        X = np.concatenate(all_sequences) if all_sequences else np.array([]).reshape(0, self.sequence_length, 0)
        y = np.concatenate(all_labels) if all_labels else np.array([])

        print(f"  → Generated {len(X)} sequences")

        return X, y

    def process_matches(self, matches: Union[Generator[Tuple[pd.Series, pd.DataFrame], None, None], List[Tuple[pd.Series, pd.DataFrame]]]) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Process multiple matches from generator or list into sequences.

        Calls process_match() for each match and concatenates all results.
        Automatically extracts match_id and home_team_id from match metadata.

        Args:
            matches: Generator or list yielding/containing (match, events_df) tuples where:
                - match: pd.Series with match metadata (must include 'match_id' and 'home_team_id')
                - events_df: pd.DataFrame with StatsBomb events for that match

        Returns:
            Tuple of (X, y) where:
                - X is array of all sequences with shape (n_sequences, sequence_length, ~38)
                - y is array of all labels with shape (n_sequences,)

        Example:
            >>> from src.data_loader import load_statsbomb_socceraction_data
            >>> from src.data_splitter import split_matches
            >>> preprocessor = SequencePreprocessor(sequence_length=10)
            >>>
            >>> # Using generator
            >>> matches_gen = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
            >>> X, y = preprocessor.process_matches(matches_gen)
            >>>
            >>> # Using list from split_matches
            >>> matches_gen = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
            >>> train_matches, val_matches, test_matches = split_matches(matches_gen)
            >>> X_train, y_train = preprocessor.process_matches(train_matches)
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
