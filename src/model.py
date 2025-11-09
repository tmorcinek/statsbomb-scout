"""Module for building sequence models (LSTM or Transformer)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LSTMSequenceModel:
    """LSTM-based model for sequence value prediction."""

    def __init__(self, input_shape: tuple, lstm_units: int = 128,
                 lstm_dropout: float = 0.2):
        """
        Initialize LSTM model.

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            lstm_units: Number of LSTM units
            lstm_dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = lstm_dropout
        self.model = None

    def build(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)

        # LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(self.dropout)(x)
        x = layers.LSTM(self.lstm_units // 2)(x)
        x = layers.Dropout(self.dropout)(x)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation='linear')(x)  # Regression output

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model.

        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )


class TransformerSequenceModel:
    """Transformer-based model for sequence value prediction."""

    def __init__(self, input_shape: tuple, num_heads: int = 4,
                 d_model: int = 128, ff_dim: int = 512,
                 num_blocks: int = 2, dropout: float = 0.1):
        """
        Initialize Transformer model.

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            num_heads: Number of attention heads
            d_model: Dimension of model
            ff_dim: Dimension of feed-forward network
            num_blocks: Number of transformer blocks
            dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.model = None

    def transformer_encoder(self, inputs):
        """Create a transformer encoder block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model
        )(inputs, inputs)
        attention_output = layers.Dropout(self.dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(
            inputs + attention_output
        )

        # Feed-forward network
        ff_output = layers.Dense(self.ff_dim, activation='relu')(attention_output)
        ff_output = layers.Dense(self.d_model)(ff_output)
        ff_output = layers.Dropout(self.dropout)(ff_output)
        output = layers.LayerNormalization(epsilon=1e-6)(
            attention_output + ff_output
        )

        return output

    def build(self) -> keras.Model:
        """
        Build Transformer model architecture.

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)

        # Project to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)

        # Positional encoding (simple learned approach)
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.input_shape[0], output_dim=self.d_model
        )(positions)
        x = x + position_embedding

        # Stack transformer blocks
        for _ in range(self.num_blocks):
            x = self.transformer_encoder(x)

        # Aggregate sequence (mean pooling)
        x = layers.GlobalAveragePooling1D()(x)

        # Output layer
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation='linear')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model.

        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )


def create_model(model_type: str, input_shape: tuple, **kwargs) -> keras.Model:
    """
    Factory function to create a model.

    Args:
        model_type: 'lstm', 'transformer', or 'attention_lstm'
        input_shape: Shape of input data
        **kwargs: Additional parameters for model

    Returns:
        Compiled Keras model
    """
    if model_type == 'lstm':
        model_builder = LSTMSequenceModel(input_shape, **kwargs)
    elif model_type == 'transformer':
        model_builder = TransformerSequenceModel(input_shape, **kwargs)
    elif model_type == 'attention_lstm':
        # Import here to avoid circular dependency
        from src.models.attention_lstm import AttentionLSTMModel
        model_builder = AttentionLSTMModel(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm', 'transformer', or 'attention_lstm'")

    model = model_builder.build()
    model_builder.compile()

    return model
