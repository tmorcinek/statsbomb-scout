"""Main script to run the full pipeline."""

import config
import pandas as pd

from src.ml.data_loader import load_statsbomb_socceraction_data
from src.ml.data_splitter import split_matches
from src.ml.preprocessing import SequencePreprocessor
from src.ml.model import create_model
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

    # Use appropriate parameters based on model type
    if config.MODEL_TYPE == 'lstm':
        model = create_model(
            model_type='lstm',
            input_shape=input_shape,
            lstm_units=config.LSTM_UNITS,
            lstm_dropout=config.LSTM_DROPOUT
        )
    elif config.MODEL_TYPE == 'transformer':
        model = create_model(
            model_type='transformer',
            input_shape=input_shape,
            num_heads=config.TRANSFORMER_HEADS,
            d_model=config.TRANSFORMER_DIM,
            ff_dim=config.TRANSFORMER_FF_DIM,
            num_blocks=config.TRANSFORMER_BLOCKS,
            dropout=config.LSTM_DROPOUT
        )
    elif config.MODEL_TYPE == 'attention_lstm':
        model = create_model(
            model_type='attention_lstm',
            input_shape=input_shape,
            lstm_units=config.LSTM_UNITS,
            lstm_dropout=config.LSTM_DROPOUT,
            return_attention=True
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}. Use 'lstm', 'transformer', or 'attention_lstm'")

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
    print(f"Best model saved as: models/best_model.h5")
    print("=" * 50)


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    main()
