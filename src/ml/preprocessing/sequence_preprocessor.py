"""Module for preprocessing event data into sequences."""

import warnings
from enum import Enum
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import socceraction.spadl.config as spadl_config
from socceraction.xthreat import ExpectedThreat

from src.ml.preprocessing.action_valuation import calculate_xt_values, get_actions_value
from src.ml.preprocessing.possessions_extraction import extract_possessions
from src.ml.preprocessing.xthreat import get_default_xt_model

warnings.filterwarnings('ignore', category=FutureWarning, module='socceraction')
pd.set_option('future.no_silent_downcasting', True)


class PreprocessingMode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    PLAYER_EVALUATION = "player_evaluation"


class SequencePreprocessor:

    def __init__(self, sequence_length: int, minimum_sequence_length: int, xt_model: ExpectedThreat | None = None):
        self.sequence_length = sequence_length
        self.minimum_sequence_length = minimum_sequence_length
        self.xt_model = xt_model or get_default_xt_model()

    def _extract_actions(self, game: pd.Series, events_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        return {
            pid: df
            for pid, df in extract_possessions(game, events_df).items() if len(df) >= self.minimum_sequence_length
        }

    def _update_action(self, action: pd.DataFrame) -> pd.DataFrame:
        """Add computed features to actions DataFrame."""
        action['xT'] = calculate_xt_values(action, self.xt_model)

        action["dx"] = action["end_x"] - action["start_x"]
        action["dy"] = action["end_y"] - action["start_y"]
        action["distance"] = np.sqrt(action["dx"] ** 2 + action["dy"] ** 2)
        action["angle"] = np.arctan2(action["dy"], action["dx"])

        action["time_diff"] = action["time_seconds"].diff().fillna(0.0)
        action["opposite_action"] = (action["team_id"] != action["possession_team_id"]).astype(int)

        return action

    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize features and convert to numpy array with one-hot encoding.

        Features (total 46):
        - Spatial (4): start_x, start_y, end_x, end_y normalized to [0,1]
        - Geometric (3): distance, sin(angle), cos(angle)
        - Temporal (1): time_diff capped at 10s
        - Contextual (3): under_pressure, counterpress, opposite_action
        - Categorical (35): one-hot encoded
          - type_id (23): action type
          - result_id (6): result type
          - bodypart_id (6): body part used

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Normalized feature array of shape (n_actions, 46)
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
        contextual = features_df[['under_pressure', 'counterpress', 'opposite_action']].astype(float).values

        # 5. Categorical features (one-hot encoding)
        def one_hot_numpy(category_ids, num_categories, dtype=np.uint8):
            a = np.asarray(category_ids)

            valid = (~np.isnan(a)) if a.dtype.kind == "f" else (a >= 0)
            idx = np.where(valid, a, -1).astype(np.int64)

            out = np.zeros((len(a), num_categories), dtype=dtype)
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

    def _pad_sequence(self, features: np.ndarray) -> np.ndarray:
        n_actions, n_features = features.shape

        if n_actions >= self.sequence_length:
            # Take last sequence_length actions
            return features[-self.sequence_length:]
        else:
            # Pad with zeros at the end
            padded = np.zeros((self.sequence_length, n_features), dtype=np.float32)
            padded[:n_actions] = features  # Copy actual data at the beginning
            return padded

    def _create_sequences(self, features: np.ndarray) -> list[tuple[np.ndarray, int]]:
        n_actions = len(features)

        if n_actions < self.sequence_length:
            return [(self._pad_sequence(features), n_actions)]

        sequences = []
        for i in range(n_actions - self.sequence_length + 1):
            end_idx = i + self.sequence_length
            sequences.append((features[i:end_idx], end_idx))

        return sequences

    def _create_validation_sequences(self, features: np.ndarray) -> list[tuple[np.ndarray, int]]:
        n_actions = len(features)
        return [(self._pad_sequence(features), n_actions)]

    def _create_evaluation_sequences(self, features: np.ndarray) -> list[tuple[np.ndarray, int]]:
        n_actions = len(features)
        sequences = []

        while n_actions >= self.sequence_length:
            sequences.append((features[n_actions - self.sequence_length:n_actions], n_actions))
            n_actions -= self.sequence_length

        if n_actions >= self.minimum_sequence_length:
            sequences.append((self._pad_sequence(features[:n_actions]), n_actions))

        return sequences

    def _create_label(self, actions_df: pd.DataFrame) -> float:
        return get_actions_value(actions_df)

    def process_match(self, match_id: int, match: pd.Series, events_df: pd.DataFrame, mode: PreprocessingMode = PreprocessingMode.TRAINING) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        print(f"→Processing match {match_id}: {len(events_df)} events (mode: {mode.value})")

        sequence_creators = {
            PreprocessingMode.TRAINING: self._create_sequences,
            PreprocessingMode.VALIDATION: self._create_validation_sequences,
            PreprocessingMode.PLAYER_EVALUATION: self._create_evaluation_sequences
        }

        all_sequences = []
        labels = []
        sequence_windows = []

        for pid, actions_df in self._extract_actions(match, events_df).items():
            features = self._update_action(actions_df)
            normalized_features = self._normalize_features(features)

            sequences = sequence_creators[mode](normalized_features)

            for seq, end_idx in sequences:
                window_actions = features.iloc[:end_idx].tail(self.sequence_length)
                labels.append(self._create_label(window_actions))
                sequence_windows.append(window_actions)

            all_sequences.append(np.array([seq for seq, _ in sequences], dtype=np.float32))

        X = np.concatenate(all_sequences)
        y = np.array(labels)
        p = np.array(sequence_windows, dtype=object)
        print(f" → Generated {len(X)} sequences, {len(y)} labels for match {match_id}")

        return X, y, p

    def process_matches(self, matches: Iterable[Tuple[pd.Series, pd.DataFrame]], mode: PreprocessingMode = PreprocessingMode.TRAINING) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        results = [
            (X, y, p, np.full((len(p),), m["game_id"], dtype=np.int64))
            for m, e in matches
            for X, y, p in [self.process_match(m["game_id"], m, e, mode=mode)]
        ]
        X, y, p, m = map(np.concatenate, zip(*results))

        print(f"\nTotal sequences from all matches: {len(X)} (mode: {mode.value})")
        return X, y, p, m
