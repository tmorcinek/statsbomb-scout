"""Test script to verify preprocessing flow."""

import numpy as np
from src.preprocessing import SequencePreprocessor
from src.data_loader import load_statsbomb_socceraction_data

def test_preprocessing_flow():
    """Test the complete preprocessing pipeline."""

    # Load one match
    print("Loading data...")
    match, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))

    print(f"\nMatch columns: {list(match.index)}")
    print(f"Match info:\n{match}")
    print(f"\nTotal events: {len(events)}")

    # Extract match_id and home_team_id
    match_id = match.get('game_id', match.name)  # Try game_id first, fallback to index
    home_team_id = match.get('home_team_id')

    print(f"\nMatch ID: {match_id}")
    print(f"Home team ID: {home_team_id}")

    # Initialize preprocessor
    preprocessor = SequencePreprocessor(sequence_length=10)

    # Test _extract_possessions
    print("\n--- Testing _extract_possessions ---")
    possessions = preprocessor._extract_possessions(events)
    print(f"Extracted {len(possessions)} possessions")

    if possessions:
        print(f"First possession length: {len(possessions[0])} events")
        print(f"Possession lengths: min={min(len(p) for p in possessions)}, "
              f"max={max(len(p) for p in possessions)}, "
              f"mean={np.mean([len(p) for p in possessions]):.1f}")

    # Test _extract_features
    print("\n--- Testing _extract_features ---")
    if possessions:
        features_df = preprocessor._extract_features(possessions[0], home_team_id)
        print(f"Features DataFrame shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        print(f"First row features:\n{features_df.iloc[0][['start_x', 'start_y', 'distance', 'angle', 'time_diff']]}")

    # Test _normalize_features
    print("\n--- Testing _normalize_features ---")
    if possessions:
        normalized = preprocessor._normalize_features(features_df)
        print(f"Normalized features shape: {normalized.shape}")
        print(f"Feature range: min={normalized.min():.3f}, max={normalized.max():.3f}")
        print(f"First action features (first 10): {normalized[0][:10]}")

    # Test _create_sequences
    print("\n--- Testing _create_sequences ---")
    if possessions:
        sequences = preprocessor._create_sequences(normalized)
        print(f"Sequences shape: {sequences.shape}")
        print(f"Expected: ({len(normalized) - preprocessor.sequence_length + 1}, {preprocessor.sequence_length}, {normalized.shape[1]})")

        if len(sequences) > 0:
            print(f"First sequence shape: {sequences[0].shape}")
            print(f"First sequence first action (first 5 features): {sequences[0][0][:5]}")

    # Test _create_labels
    print("\n--- Testing _create_labels ---")
    if possessions and len(sequences) > 0:
        labels = preprocessor._create_labels(features_df, list(range(len(sequences))))
        print(f"Labels shape: {labels.shape}")
        print(f"Labels (first 5): {labels[:5]}")

    # Test full pipeline
    print("\n--- Testing process_match ---")
    X, y = preprocessor.process_match(match_id, home_team_id, events)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

    # Validate shapes
    print("\n--- Validation ---")
    assert X.shape[0] == y.shape[0], "Number of sequences and labels must match!"
    assert X.shape[1] == preprocessor.sequence_length, f"Sequence length must be {preprocessor.sequence_length}!"
    assert X.shape[2] > 0, "Feature dimension must be > 0!"
    print("✅ All validations passed!")

    return X, y

if __name__ == "__main__":
    X, y = test_preprocessing_flow()
    print(f"\n🎉 Preprocessing flow works correctly!")
    print(f"Generated {len(X)} sequences with {X.shape[2]} features each")

