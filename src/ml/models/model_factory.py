from pathlib import Path

from tensorflow import keras

import config
from src.ml.models.lstm import LSTMSequenceModel
from src.ml.models.transformer import TransformerSequenceModel


def create_model(model_type: str, input_shape: tuple, **kwargs) -> keras.Model:
    if model_type == 'lstm':
        model_builder = LSTMSequenceModel(
            input_shape,
            lstm_units=kwargs.get('lstm_units', config.LSTM_UNITS),
            dropout=kwargs.get('dropout', config.DROPOUT),
            use_second_lstm=kwargs.get('use_second_lstm', True)
        )
    elif model_type == 'transformer':
        model_builder = TransformerSequenceModel(
            input_shape,
            num_heads=kwargs.get('num_heads', config.TRANSFORMER_HEADS),
            d_model=kwargs.get('d_model', config.TRANSFORMER_DIM),
            ff_dim=kwargs.get('ff_dim', config.TRANSFORMER_FF_DIM),
            num_blocks=kwargs.get('num_blocks', config.TRANSFORMER_BLOCKS),
            dropout=kwargs.get('dropout', config.DROPOUT)
        )
    elif model_type == 'attention_lstm':
        from src.ml.models.attention_lstm import AttentionLSTMModel
        model_builder = AttentionLSTMModel(
            input_shape,
            lstm_units=kwargs.get('lstm_units', config.LSTM_UNITS),
            dropout=kwargs.get('dropout', config.DROPOUT)
        )
    elif model_type == 'bigru':
        from src.ml.models.bigru import build_seq_value_model
        model = build_seq_value_model(
            input_shape,
            rnn_units=kwargs.get('gru_units', config.LSTM_UNITS),
            attn_hidden=kwargs.get('attn_hidden', 64),
            dropout=kwargs.get('dropout', config.DROPOUT),
            recurrent_dropout=kwargs.get('recurrent_dropout', 0.15),
            l2_reg=kwargs.get('l2_reg', 0.01),
            return_attention=False
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=kwargs.get('learning_rate', 0.0005),
                clipnorm=1.0
            ),
            loss='mse',
            metrics=['mae']
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm', 'transformer', 'attention_lstm', or 'bigru'")

    model = model_builder.build()
    model_builder.compile(learning_rate=kwargs.get('learning_rate', 0.001))

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
        from src.ml.models.transformer import MultiHeadAttentionWithWeights, AddPositionalEncoding
        custom_objects['MultiHeadAttentionWithWeights'] = MultiHeadAttentionWithWeights
        custom_objects['AddPositionalEncoding'] = AddPositionalEncoding
    elif model_type == 'bigru':
        from src.ml.models.bigru import TemporalAttentionPooling, build_seq_value_model
        custom_objects['TemporalAttentionPooling'] = TemporalAttentionPooling
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        input_shape = model.input_shape[1:]

        model_with_attn = build_seq_value_model(
            input_shape=input_shape,
            rnn_units=model.get_layer('bigru').forward_layer.units,
            attn_hidden=64,
            dropout=0.0,
            return_attention=True
        )
        model_with_attn.set_weights(model.get_weights())
        return model_with_attn

    if custom_objects:
        return keras.models.load_model(model_path, custom_objects=custom_objects)

    return keras.models.load_model(model_path)
