from pathlib import Path

from tensorflow import keras

import config
from src.ml.models.lstm import LSTMSequenceModel
from src.ml.models.transformer import TransformerSequenceModel


def create_model(model_type: str, input_shape: tuple) -> keras.Model:
    if model_type == 'lstm':
        model_builder = LSTMSequenceModel(
            input_shape,
            lstm_units=config.LSTM_UNITS,
            dropout=config.DROPOUT
        )
    elif model_type == 'transformer':
        model_builder = TransformerSequenceModel(
            input_shape,
            num_heads=config.TRANSFORMER_HEADS,
            d_model=config.TRANSFORMER_DIM,
            ff_dim=config.TRANSFORMER_FF_DIM,
            num_blocks=config.TRANSFORMER_BLOCKS,
            dropout=config.DROPOUT
        )
    elif model_type == 'attention_lstm':
        from src.ml.models.attention_lstm import AttentionLSTMModel
        model_builder = AttentionLSTMModel(
            input_shape,
            lstm_units=config.LSTM_UNITS,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm', 'transformer', or 'attention_lstm'")

    model = model_builder.build()
    model_builder.compile()

    return model


def load_model(model_type: str, model_dir: str = None) -> keras.Model:
    if model_dir is None:
        model_dir = f"models/{model_type}/"

    model_path = Path(model_dir) / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    custom_objects = {}

    if model_type == 'attention_lstm':
        from src.ml.models.attention_lstm import AttentionLayer
        custom_objects['AttentionLayer'] = AttentionLayer
    elif model_type == 'transformer':
        from src.ml.models.transformer import AttentionWeightsLayer, AddPositionalEncoding
        custom_objects['AttentionWeightsLayer'] = AttentionWeightsLayer
        custom_objects['AddPositionalEncoding'] = AddPositionalEncoding

    if custom_objects:
        return keras.models.load_model(model_path, custom_objects=custom_objects)

    return keras.models.load_model(model_path)
