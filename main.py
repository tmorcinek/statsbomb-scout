"""Main script to run the full pipeline."""

import time

import pandas as pd

import config
from src.data.data_loader import load_statsbomb_socceraction_data
from src.data.data_splitter import split_matches
from src.ml.models.model_factory import create_model
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.train import ModelTrainer
from src.ml.xthreat import get_default_xt_model

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    print("=" * 50)
    print("Football Player Evaluation Model")
    print("=" * 50)

    print("\n1a. Loading data...")
    start_time = time.time()
    loading_start = time.time()
    data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)

    print("\n1b. Splitting data...")
    train_matches, val_matches, test_matches = split_matches(data)
    loading_time = time.time() - loading_start
    print(f"⏱️  Data loading time: {loading_time:.2f}s")

    print("\n2. Preprocessing data...")
    preprocessing_start = time.time()
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    X_train, y_train, p_train, m_train = preprocessor.process_matches(train_matches)
    X_val, y_val, p_val, m_val = preprocessor.process_matches(val_matches)
    X_test, y_test, p_test, m_test = preprocessor.process_matches(test_matches)
    preprocessing_time = time.time() - preprocessing_start
    print(f"⏱️  Data preprocessing time: {preprocessing_time:.2f}s")

    print(f"Train set: {X_train.shape[0]} sequences")
    print(f"Val set: {X_val.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")

    print(f"\n3. Building {config.MODEL_TYPE.upper()} model...")
    model = create_model(config.MODEL_TYPE, input_shape=(config.SEQUENCE_LENGTH, X_train.shape[2]))
    model.summary()

    print("\n4. Training model...")
    training_start = time.time()
    trainer = ModelTrainer(model)
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS
    )
    training_time = time.time() - training_start
    print(f"⏱️  Model training time: {training_time:.2f}s")

    print("\n5. Evaluating model...")
    val_metrics = trainer.evaluate(X_val, y_val)
    test_metrics = trainer.evaluate(X_test, y_test)

    all_metrics = {
        'validation': val_metrics,
        'test': test_metrics
    }

    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"\n⏱️  Total execution time: {int(minutes)}m {seconds:.2f}s")

    print("\n6. Saving results...")
    trainer.save_training_metrics(all_metrics)
    trainer.plot_training_history(filename="training_history.png")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print(f"Best model saved as: models/{config.MODEL_TYPE}/best_model.keras")
    print("=" * 50)
