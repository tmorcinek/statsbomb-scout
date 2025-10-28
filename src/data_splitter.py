"""Data splitting utilities for train/val/test splits."""

import random
from typing import Generator, List, Tuple

import pandas as pd


def split_matches(
        matches_generator: Generator[Tuple[int, pd.DataFrame], None, None],
        test_size: float = 0.1,
        val_size: float = 0.2,
        random_seed: int = 42
) -> Tuple[List[Tuple[int, pd.DataFrame]], ...]:
    """
    Split matches into train, validation, and test sets.

    This ensures no data leakage - sequences from the same match
    won't appear in both training and test/validation sets.

    Args:
        matches_generator: Generator yielding (match_id, events_df) tuples
        test_size: Proportion for test set (default: 0.1 = 10%)
        val_size: Proportion of remaining matches for validation (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_matches, val_matches, test_matches)
        Each is a list of (match_id, events_df) tuples
    """
    # Convert generator to list
    all_matches = list(matches_generator)
    n_matches = len(all_matches)

    print(f"Splitting {n_matches} matches...")

    # Calculate split indices
    n_test = int(n_matches * test_size)
    n_val = int((n_matches - n_test) * val_size)

    # Shuffle matches for random split
    random.seed(random_seed)
    random.shuffle(all_matches)

    # Split
    test_matches = all_matches[:n_test]
    val_matches = all_matches[n_test:n_test + n_val]
    train_matches = all_matches[n_test + n_val:]

    print(f"  → Train: {len(train_matches)} matches ({len(train_matches) / n_matches * 100:.1f}%)")
    print(f"  → Val: {len(val_matches)} matches ({len(val_matches) / n_matches * 100:.1f}%)")
    print(f"  → Test: {len(test_matches)} matches ({len(test_matches) / n_matches * 100:.1f}%)")

    return train_matches, val_matches, test_matches
