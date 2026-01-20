"""Module for preprocessing event data into sequences."""

import warnings
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import socceraction.spadl.config as spadl_config
from socceraction.xthreat import ExpectedThreat

from src.ml.action_valuation import calculate_xt_values
from src.ml.preprocessing.possessions_extraction import extract_possessions
from src.ml.xthreat import get_default_xt_model

warnings.filterwarnings('ignore', category=FutureWarning, module='socceraction')
pd.set_option('future.no_silent_downcasting', True)


class SequencePreprocessor:
    """Preprocesses event data into fixed-length sequences with features."""

    def __init__(self, sequence_length: int, xt_model: ExpectedThreat | None = None):
        self.sequence_length = sequence_length
        self.xt_model = xt_model or get_default_xt_model()
        self.action_type_mapping = {}

    def _extract_actions(self, game: pd.Series, events_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        return {
            pid: df
            for pid, df in extract_possessions(game, events_df).items() if len(df) >= self.sequence_length
        }

    def _update_action(self, action: pd.DataFrame) -> pd.DataFrame:
        """Add computed features to actions DataFrame."""
        action['xT'] = calculate_xt_values(action, self.xt_model)

        action["dx"] = action["end_x"] - action["start_x"]
        action["dy"] = action["end_y"] - action["start_y"]
        action["distance"] = np.sqrt(action["dx"] ** 2 + action["dy"] ** 2)
        action["angle"] = np.arctan2(action["dy"], action["dx"])

        action["time_diff"] = action["time_seconds"].diff().fillna(0.0)

        return action

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

    def _create_label(self, actions_df: pd.DataFrame) -> float:
        """Create label for possession by comparing total xG and xT."""
        total_xg = actions_df['xG'].sum()
        total_xt = actions_df['xT'].sum()
        return max(total_xg, total_xt)

    def process_match(self, match_id: int, match: pd.Series, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"→Processing match {match_id}: {len(events_df)} events")

        all_sequences = []
        labels = []
        possession_ids = []

        for pid, actions_df in self._extract_actions(match, events_df).items():
            features = self._update_action(actions_df)

            normalized_features = self._normalize_features(features)

            sequence = self._create_simple_sequence(normalized_features)

            label = self._create_label(features.tail(self.sequence_length))

            all_sequences.append(sequence)
            labels.append(label)
            possession_ids.append(pid)

        X = np.concatenate(all_sequences)
        y = np.array(labels)
        p = np.array(possession_ids)
        print(f" → Generated {len(X)} sequences, {len(y)} labels for match {match_id}")

        return X, y, p

    def process_matches(self, matches: Iterable[Tuple[pd.Series, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        results = [
            (X, y, p,  np.full_like(p, m["game_id"]))
            for m, e in matches
            for X, y, p in [self.process_match(m["game_id"], m, e)]
        ]
        X, y, p, m = map(np.concatenate, zip(*results))

        print(f"\nTotal sequences from all matches: {len(X)}")
        return X, y, p, m
