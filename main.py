"""Main script to run the full pipeline."""
from setuptools.dist import sequence

import config
import pandas as pd

from src.data_loader import load_statsbomb_data, get_home_team_id, load_statsbomb_socceraction_data
from src.data_splitter import split_matches
from src.preprocessing import SequencePreprocessor
from src.model import create_model
from src.train import ModelTrainer
import inspect
from socceraction.data import statsbomb
import socceraction.spadl as spadl

from src.xthreat import get_default_xt_model


def main():
    """Run the full training pipeline."""
    print("=" * 50)
    print("Football Player Evaluation Model")
    print("=" * 50)

    # 1a. Load data
    print("\n1a. Loading data...")
    # data = load_statsbomb_data( 55, 282)
    data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)

    # 1b. Splitting data
    print("\n1b. Splitting data...")
    train_matches, val_matches, test_matches = split_matches(data)

    # 2. Preprocess data
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    X_train, y_train = preprocessor.process_matches(train_matches)
    X_val, y_val = preprocessor.process_matches(val_matches)
    X_test, y_test = preprocessor.process_matches(test_matches)

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

def single_match_pipeline():
    # Load events for the match
    # match, events = next(load_statsbomb_data(55, 282))
    match, events = next(load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282))

    # match, events = next(load_statsbomb_data(55, 282))
    # Preprocess data
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    possessions = preprocessor._extract_possessions(events)
    possession = next((x for x in possessions if len(x) in range(17, 20)), None)
    # possession = next((x for x in possessions if x.iloc[0]["possession"] == 2), None)
    # possession = next((x for x in possessions if len(x) in [11..14]), None)
    # possession = possessions[1]
    print(f"Possession: \n{possession}" )
    features_df = preprocessor._extract_features(possession.copy(), match["home_team_id"])
    print(f"Extracted features: \n {features_df}" )

    normalized_features = preprocessor._normalize_features(features_df)
    sequences = preprocessor._create_sequences(normalized_features)

    # print(normalized_features)
    print(normalized_features.shape)
    print(len(sequences))
    print(list(range(len(sequences))))
    # features = spadl.statsbomb.convert_to_actions(possession.copy(), home_team_id)

    # first_features = preprocessor._create_features(first)
    # third_features = preprocessor._create_features(third)


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # main()
    single_match_pipeline()
