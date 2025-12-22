"""Module for preprocessing event data into sequences."""

import warnings
from typing import List, Tuple, Generator, Union, Iterable

import numpy as np
import pandas as pd
import socceraction.spadl as spadl
import socceraction.spadl.config as spadl_config
from socceraction.xthreat import ExpectedThreat

from src.ml.action_valuation import calculate_xt_values, calculate_xg_values
from src.ml.xthreat import get_default_xt_model

warnings.filterwarnings('ignore', category=FutureWarning, module='socceraction')
pd.set_option('future.no_silent_downcasting', True)


class SequencePreprocessor:
    """Preprocesses event data into fixed-length sequences with features."""

    def __init__(self, sequence_length: int, xt_model: ExpectedThreat = get_default_xt_model()):
        self.sequence_length = sequence_length
        self.xt_model = xt_model
        self.action_type_mapping = {}

    def _extract_possessions(self, events_df: pd.DataFrame) -> List[pd.DataFrame]:
        return [
            possession_group
            for possession_id, possession_group in events_df.groupby("possession") if len(possession_group) >= self.sequence_length
        ]

    def _extract_features(self, possession: pd.DataFrame) -> pd.DataFrame:
        """Extract features from possession events, converting to SPADL and adding computed features."""
        actions = spadl.statsbomb.convert_to_actions(possession, possession.iloc[0]['team_id'], xy_fidelity_version=2)

        actions["dx"] = actions["end_x"] - actions["start_x"]
        actions["dy"] = actions["end_y"] - actions["start_y"]
        actions["distance"] = np.sqrt(actions["dx"] ** 2 + actions["dy"] ** 2)
        actions["angle"] = np.arctan2(actions["dy"], actions["dx"])

        actions["time_diff"] = actions["time_seconds"].diff().fillna(0.0)

        subset = possession[['event_id', 'duration', 'under_pressure', 'counterpress', 'possession']].copy()
        subset['xG'] = calculate_xg_values(possession)
        actions = actions.merge(subset, left_on='original_event_id', right_on='event_id', how='left').drop(columns='event_id')
        actions = actions.fillna({'duration': 0.0, 'under_pressure': False, 'counterpress': False, 'xG': 0.0})

        actions['xT'] = calculate_xt_values(actions, self.xt_model)

        return actions

    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize features and convert to numpy array with one-hot encoding.

        Features (total ~38):
        - Spatial (4): start_x, start_y, end_x, end_y normalized to [0,1]
        - Geometric (3): distance, sin(angle), cos(angle)
        - Temporal (1): time_diff capped at 10s
        - Contextual (2): under_pressure, counterpress
        - Categorical (~28): one-hot encoded type_id, result_id, bodypart_id

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Normalized feature array of shape (n_actions, ~38)
        """
        # One-hot encoding dimensions
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

        type_onehot = one_hot_numpy(features_df["type_id"], n_types)
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
        """Create overlapping sequences using sliding window, returning shape (n_sequences, sequence_length, n_features)."""
        if len(features) < self.sequence_length:
            raise ValueError(f"Insufficient number of actions ({len(features)}) for sequence length {self.sequence_length}.")

        return np.lib.stride_tricks.sliding_window_view(features, (self.sequence_length, features.shape[1])).squeeze(1)

    def _create_simple_sequence(self, features: np.ndarray) -> np.ndarray:
        """Create single sequence from last N actions, returning shape (1, sequence_length, n_features)."""
        if len(features) < self.sequence_length:
            raise ValueError(f"Insufficient number of actions ({len(features)}) for sequence length {self.sequence_length}.")

        return features[-self.sequence_length:].reshape(1, self.sequence_length, -1)

    def _create_label(self, actions_df: pd.DataFrame) -> np.ndarray:
        """Create label for possession by comparing total xG and xT."""
        total_xg = actions_df['xG'].sum()
        total_xt = actions_df['xT'].sum()
        return np.array([max(total_xg, total_xt)])

    def process_match(self, match_id: int, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print(f"→Processing match {match_id}: {len(events_df)} events")

        possessions = self._extract_possessions(events_df)

        all_sequences = []
        all_labels = []

        for possession in possessions:
            features = self._extract_features(possession)
            if len(features) < self.sequence_length:
                continue

            normalized_features = self._normalize_features(features)

            sequence = self._create_simple_sequence(normalized_features)

            labels = self._create_label(features.tail(self.sequence_length))

            all_sequences.append(sequence)
            all_labels.append(labels)

        X = np.concatenate(all_sequences) if all_sequences else np.array([]).reshape(0, self.sequence_length, 0)
        y = np.concatenate(all_labels) if all_labels else np.array([])

        print(f" → Generated {len(X)} sequences, {len(y)} labels for match {match_id}")

        return X, y

    def process_matches(self, matches: Iterable[Tuple[pd.Series, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray]:
        all_X, all_y = zip(*(self.process_match(match["game_id"], events_df) for match, events_df in matches))

        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        print(f"\nTotal sequences from all matches: {len(X)}")
        return X, y
