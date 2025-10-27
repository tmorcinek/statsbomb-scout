"""Main script to run the full pipeline."""

import config
from src.preprocessing import SequencePreprocessor
from socceraction.data.statsbomb import StatsBombLoader
from src.model import create_model
from src.train import ModelTrainer


def main():
    """Run the full training pipeline."""
    print("=" * 50)
    print("Football Player Evaluation Model")
    print("=" * 50)

    # 1. Load data
    print("\n1. Loading data...")
    SBL = StatsBombLoader(root="data/statsbomb/data", getter="local")
    df_competitions = SBL.competitions()
    print(df_competitions)

    # TODO: Specify your data file
    # events_df = data_loader.load_from_json("path/to/events.json")
    # events_df = data_loader.filter_relevant_events()

    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH)
    # TODO: Uncomment when data is loaded
    # X, y = preprocessor.prepare_dataset(events_df)
    # X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    #     X, y, test_size=config.TEST_SPLIT, val_size=config.VALIDATION_SPLIT
    # )

    # 3. Build model
    print(f"\n3. Building {config.MODEL_TYPE.upper()} model...")
    # TODO: Get input shape from data
    # input_shape = (config.SEQUENCE_LENGTH, X.shape[2])
    # model = create_model(
    #     model_type=config.MODEL_TYPE,
    #     input_shape=input_shape,
    #     lstm_units=config.LSTM_UNITS,
    #     dropout=config.LSTM_DROPOUT
    # )
    # model.summary()

    # 4. Train model
    print("\n4. Training model...")
    # trainer = ModelTrainer(model)
    # history = trainer.train(
    #     X_train, y_train,
    #     X_val, y_val,
    #     batch_size=config.BATCH_SIZE,
    #     epochs=config.EPOCHS
    # )

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    # metrics = trainer.evaluate(X_test, y_test)

    # 6. Save results
    print("\n6. Saving results...")
    # trainer.plot_training_history(save_path="models/training_history.png")
    # trainer.save_model("final_model.h5")
    # trainer.save_training_metrics(metrics)

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

