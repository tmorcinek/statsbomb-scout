import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerSequenceModel:
    """Transformer-based model for sequence value prediction."""

    def __init__(self, input_shape: tuple, num_heads: int = 4, d_model: int = 128, ff_dim: int = 512, num_blocks: int = 2, dropout: float = 0.1):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.model = None

    def transformer_encoder(self, inputs):
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
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
