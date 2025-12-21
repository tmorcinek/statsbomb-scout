"""Script to analyze actions in a specific match using the trained attention_lstm model."""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import config
from src.data.data_loader import load_socceraction_match
from src.ml.models.attention_lstm import AttentionLayer
from src.ml.preprocessing.sequence import SequencePreprocessor
from src.ml.xthreat import get_default_xt_model


def analyze_match_actions(match_id: int, competition_id: int = 55, season_id: int = 282, model_path: str = "models/attention_lstm/best_model.keras"):
    print(f"Analyzing match {match_id}...")

    print("1. Loading match data...")
    match_info, events = load_socceraction_match("data/statsbomb/data", competition_id, season_id, match_id)
    print(f"Match info: {match_info}")
    print(f"Events loaded: {len(events)}")

    # 2. Preprocess into sequences
    print("2. Preprocessing into sequences...")
    preprocessor = SequencePreprocessor(sequence_length=config.SEQUENCE_LENGTH, xt_model=get_default_xt_model())
    X_sequences, y_true = preprocessor.process_match(match_id, match_info['home_team_id'], events)

    if len(X_sequences) == 0:
        print("No valid sequences found in this match.")
        return

    print(f"Created {len(X_sequences)} sequences from {len(events)} events")

    # 3. Load trained model
    print("3. Loading trained model...")
    model = keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    print("Model loaded successfully")

    # 4. Make predictions
    print("4. Making predictions...")
    predictions = model.predict(X_sequences, batch_size=32, verbose=1)

    # Extract values and attention weights
    predicted_values = predictions['value'].flatten()
    attention_weights = predictions['attention_weights']  # Shape: (n_sequences, sequence_length)

    # 5. Analyze and display results
    print("\n5. Analysis Results:")
    print("=" * 50)

    # Overall match statistics
    print(f"Total sequences analyzed: {len(predicted_values)}")
    print(f"Sum of all predicted values: {predicted_values.sum():.3f}")

    # Find sequences with highest predicted values
    top_indices = np.argsort(predicted_values)[-5:][::-1]  # Top 5
    print(f"\nTop 5 sequences with highest predicted values:")
    for i, idx in enumerate(top_indices):
        seq_value = predicted_values[idx]
        seq_weights = attention_weights[idx]
        max_weight_idx = np.argmax(seq_weights)

        print(f"{i + 1}. Sequence {idx}: Value = {seq_value:.3f}")
        print(f"   Attention weights: {seq_weights}")
        print(f"   Most important action in sequence: position {max_weight_idx} (weight: {seq_weights[max_weight_idx]:.3f})")
        print()

    # Visualize attention for top sequence
    if len(top_indices) > 0:
        plot_attention_weights(attention_weights[top_indices[0]], sequence_idx=top_indices[0], value=predicted_values[top_indices[0]])

    return {
        'match_info': match_info,
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

    # You can further analyze results here
    # For example, correlate with actual match events, player analysis, etc.
