"""Script to analyze actions in a specific match using the trained attention_lstm model."""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import config
from src.analysis.game_utils import game_summary
from src.analysis.visualization import plot_multiple_possessions, _title_with_value
from src.data.data_loader import load_socceraction_match
from src.ml.models.attention_lstm import AttentionLayer
from src.ml.preprocessing.possessions_extraction import extract_possessions, tail_dataframes_to_sequence_length
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.xthreat import get_default_xt_model


def analyze_match_actions(match_id: int, competition_id: int = 55, season_id: int = 282, model_path: str = "models/attention_lstm/best_model.keras",
                          number_of_top_predictions: int = 8):
    print(f"Analyzing match with id = {match_id}...")

    print("\n1. Loading match data...")
    match, events = load_socceraction_match("data/statsbomb/data", competition_id, season_id, match_id)
    print(f"Match info: {game_summary(match)}")
    print(f"Events loaded: {len(events)}")

    print("\n2. Preprocessing into sequences...")
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    X_sequences, y_true, p_ids = preprocessor.process_match(match_id, match, events)

    print(f"Created {len(X_sequences)} sequences from {len(events)} events")

    print("\n3. Loading trained model...")
    model = keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

    print("\n4. Making predictions...")
    predictions = model.predict(X_sequences, batch_size=32, verbose=1)

    print(f"Predictions completed. Predictions type: \n{type(predictions)}, keys: {predictions.keys()}")
    predicted_values = predictions['value'].flatten()
    attention_weights = predictions['attention_weights']  # Shape: (n_sequences, sequence_length)

    print("\n5. Analysis Results:")
    print("=" * 50)

    print(f"Total sequences analyzed: {len(predicted_values)}")
    print(f"Sum of all predicted values: {predicted_values.sum():.3f}")

    top_indices = np.argsort(predicted_values)[-number_of_top_predictions:][::-1]  # Top 5
    print(f"Top {number_of_top_predictions} sequences with highest predicted values:\n")
    for i, idx in enumerate(top_indices):
        seq_value = predicted_values[idx]
        seq_weights = attention_weights[idx]
        possession_id = p_ids[idx]
        max_weight_idx = np.argmax(seq_weights)

        print(f"{i + 1}. Sequence with id = {possession_id}: Value = {seq_value:.3f}")
        print(f"   Attention weights: {seq_weights}")
        print(f"   Most important action in sequence: position {max_weight_idx} (weight: {seq_weights[max_weight_idx]:.3f})")
        print()


    print("\n6. Visualizing top possessions:")
    print("=" * 50)

    top_possessions = [extract_possessions(match, events)[p_ids[idx]] for idx in top_indices]
    top_possessions = tail_dataframes_to_sequence_length(top_possessions, config.SEQUENCE_LENGTH)

    top_possession_titles = []
    for i, idx in enumerate(top_indices):
        possession = top_possessions[i]
        predicted_value = predicted_values[idx]
        attention_weight = attention_weights[idx]

        top_possession_titles.append(_title_with_value(possession, predicted_value, attention_weight))

    fig = plot_multiple_possessions(top_possessions, titles=top_possession_titles, main_title="Top Possessions by xT Value")
    fig.savefig('data/plot/possessions_best_lstm_attention.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'match': match,
        'predicted_values': predicted_values,
        'attention_weights': attention_weights,
        'sequences': X_sequences
    }


def plot_attention_weights(weights: np.ndarray, sequence_idx: int, value: float):
    """
    Plot attention weights for a sequence.

    Args:
        weights: Attention weights array
        sequence_idx: Index of the sequence
        value: Predicted value
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights, color='skyblue')
    plt.xlabel('Action Position in Sequence')
    plt.ylabel('Attention Weight')
    plt.title('.3f')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example usage - replace with actual match ID
    MATCH_ID = 3942819  # first game Netherlands vs England Euro 2024

    results = analyze_match_actions(MATCH_ID)
