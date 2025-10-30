"""
Example notebook/script showing how to use the StatsBomb Scout pipeline.

This file demonstrates how to:
1. Load StatsBomb event data
2. Preprocess it into sequences
3. Build and train a model
4. Analyze results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import project modules
import config
from src.data_loader import load_statsbomb_socceraction_data
from src.data_splitter import split_matches
from src.preprocessing import SequencePreprocessor
from src.model import create_model
from src.train import ModelTrainer


def example_with_dummy_data():
    """
    Example showing the full pipeline with dummy data.
    Replace this with real StatsBomb data when available.
    """
    print("=" * 60)
    print("EXAMPLE: Football Player Evaluation with Dummy Data")
    print("=" * 60)

    # Create dummy data for demonstration
    print("\n1. Creating dummy event data...")
    n_events = 1000
    dummy_events = pd.DataFrame({
        'match_id': np.random.randint(1, 10, n_events),
        'type': np.random.choice(['Pass', 'Carry', 'Ball Receipt*', 'Shot'], n_events),
        'possession': np.repeat(np.arange(n_events // 10), 10),
        'x_start': np.random.uniform(0, 120, n_events),
        'y_start': np.random.uniform(0, 80, n_events),
        'x_end': np.random.uniform(0, 120, n_events),
        'y_end': np.random.uniform(0, 80, n_events),
        'timestamp': pd.date_range('2024-01-01', periods=n_events, freq='1s'),
        'under_pressure': np.random.choice([True, False], n_events),
        'player_name': np.random.choice(['Player A', 'Player B', 'Player C'], n_events),
    })

    # Add xG for shots
    dummy_events['shot_xg'] = 0.0
    shot_mask = dummy_events['type'] == 'Shot'
    dummy_events.loc[shot_mask, 'shot_xg'] = np.random.uniform(0.01, 0.8, shot_mask.sum())

    print(f"   Created {len(dummy_events)} dummy events")
    print(f"   Event types: {dummy_events['type'].value_counts().to_dict()}")

    # 2. Preprocess data into sequences
    print("\n2. Preprocessing events into sequences...")
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH)

    # For demo: create random sequences (replace with real data using load_statsbomb_socceraction_data)
    # Example real usage:
    # matches_gen = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
    # train_matches, val_matches, test_matches = split_matches(matches_gen)
    # X_train, y_train = preprocessor.process_matches(train_matches)
    # X_val, y_val = preprocessor.process_matches(val_matches)
    # X_test, y_test = preprocessor.process_matches(test_matches)

    # For now, using dummy sequences
    n_sequences = 100
    n_features = 45  # Actual number of features from preprocessing

    X = np.random.randn(n_sequences, config.SEQUENCE_LENGTH, n_features).astype(np.float32)
    y = np.random.uniform(0, 0.5, n_sequences)

    print(f"   Created {n_sequences} sequences")
    print(f"   Sequence shape: {X.shape}")
    print(f"   Label shape: {y.shape}")

    # 3. Split data
    print("\n3. Splitting data...")
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VALIDATION_SPLIT, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]} sequences")
    print(f"   Validation set: {X_val.shape[0]} sequences")
    print(f"   Test set: {X_test.shape[0]} sequences")

    # 4. Build model
    print(f"\n4. Building {config.MODEL_TYPE.upper()} model...")
    input_shape = (config.SEQUENCE_LENGTH, n_features)

    if config.MODEL_TYPE == 'lstm':
        model = create_model(
            model_type='lstm',
            input_shape=input_shape,
            lstm_units=config.LSTM_UNITS,
            lstm_dropout=config.LSTM_DROPOUT
        )
    else:
        model = create_model(
            model_type='transformer',
            input_shape=input_shape,
            num_heads=config.TRANSFORMER_HEADS,
            d_model=config.TRANSFORMER_DIM,
            ff_dim=config.TRANSFORMER_FF_DIM,
            num_blocks=config.TRANSFORMER_BLOCKS,
            dropout=0.2
        )

    print("\n   Model architecture:")
    model.summary()

    # 5. Train model
    print("\n5. Training model (with reduced epochs for demo)...")
    trainer = ModelTrainer(model, model_dir=Path("models/"))

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=config.BATCH_SIZE,
        epochs=5  # Reduced for demo
    )

    # 6. Evaluate
    print("\n6. Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)

    # 7. Save results
    print("\n7. Saving results...")
    trainer.save_model("example_model.h5")
    trainer.save_training_metrics(metrics, "example_metrics.json")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace dummy sequences with real StatsBomb data (see commented code above)")
    print("2. Adjust hyperparameters in config.py based on validation results")
    print("3. Train on full dataset with more epochs")
    print("4. Implement player/team evaluation using trained model")


def load_statsbomb_data_example():
    """
    Example of how to load real StatsBomb data.
    Uncomment and adapt when you have real data.
    """
    print("\nExample: Loading StatsBomb data with full pipeline")
    print("-" * 40)

    # Using our data loader with socceraction
    # from src.data_loader import load_statsbomb_socceraction_data
    # from src.data_splitter import split_matches
    # from src.preprocessing import SequencePreprocessor
    #
    # # Load data (competition_id=55, season_id=282 is Women's World Cup 2019)
    # matches_gen = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
    #
    # # Split into train/val/test
    # train_matches, val_matches, test_matches = split_matches(matches_gen)
    #
    # # Preprocess into sequences
    # preprocessor = SequencePreprocessor(sequence_length=10)
    # X_train, y_train = preprocessor.process_matches(train_matches)
    # X_val, y_val = preprocessor.process_matches(val_matches)
    # X_test, y_test = preprocessor.process_matches(test_matches)
    #
    # print(f"Training: {X_train.shape}")
    # print(f"Validation: {X_val.shape}")
    # print(f"Test: {X_test.shape}")

    print("(Uncomment the code above to load real data)")


if __name__ == "__main__":
    # Run the example with dummy data
    example_with_dummy_data()

    # Show how to load real data
    print("\n\n")
    load_statsbomb_data_example()

