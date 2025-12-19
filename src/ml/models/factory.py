from tensorflow import keras

from src.ml.models.lstm import LSTMSequenceModel
from src.ml.models.transformer import TransformerSequenceModel


def create_model(model_type: str, input_shape: tuple, **kwargs) -> keras.Model:
    if model_type == 'lstm':
        model_builder = LSTMSequenceModel(input_shape, **kwargs)
    elif model_type == 'transformer':
        model_builder = TransformerSequenceModel(input_shape, **kwargs)
    elif model_type == 'attention_lstm':
        # Import here to avoid circular dependency
        from src.ml.models.attention_lstm import AttentionLSTMModel
        model_builder = AttentionLSTMModel(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm', 'transformer', or 'attention_lstm'")

    model = model_builder.build()
    model_builder.compile()

    return model
