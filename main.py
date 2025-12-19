"""Main script to run the full pipeline."""

import config
import pandas as pd

from src.data.data_loader import load_statsbomb_socceraction_data
from src.data.data_splitter import split_matches
from src.ml.preprocessing import SequencePreprocessor
from src.ml.models.factory import create_model
from src.ml.train import ModelTrainer
from src.ml.xthreat import get_default_xt_model


def main():
    """Run the full training pipeline."""
    print("=" * 50)
    print("Football Player Evaluation Model")
    print("=" * 50)

    # 1a. Load data
    print("\n1a. Loading data...")
    data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)

    # 1b. Splitting data
    print("\n1b. Splitting data...")
    train_matches, val_matches, test_matches = split_matches(data)

    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    X_train, y_train = preprocessor.process_matches(train_matches)
    X_val, y_val = preprocessor.process_matches(val_matches)
    X_test, y_test = preprocessor.process_matches(test_matches)

    print(f"Train set: {X_train.shape[0]} sequences")
    print(f"Val set: {X_val.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")

    # 3. Build model
    print(f"\n3. Building {config.MODEL_TYPE.upper()} model...")
    input_shape = (config.SEQUENCE_LENGTH, X_train.shape[2])

    model = create_model(config.MODEL_TYPE, input_shape=input_shape)
    model.summary()

    # 4. Train model
    print("\n4. Training model...")
    trainer = ModelTrainer(model)
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS
    )

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)

    # 6. Save results
    print("\n6. Saving results...")
    trainer.save_training_metrics(metrics)
    trainer.plot_training_history(filename="training_history.png")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print(f"Best model saved as: models/{config.MODEL_TYPE}/best_model.keras")
    print("=" * 50)


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    main()
