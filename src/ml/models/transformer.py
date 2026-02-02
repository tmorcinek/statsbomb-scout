import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AddPositionalEncoding(layers.Layer):
    """Add learned positional encoding to input, preserving mask."""

    def __init__(self, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.supports_masking = True

    def build(self, input_shape):
        self.position_embedding = layers.Embedding(
            input_dim=self.max_seq_len,
            output_dim=self.d_model,
            name='position_embedding'
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        # Broadcasting: (seq_len, d_model) will broadcast with (batch, seq_len, d_model)
        return inputs + position_embeddings

    def compute_mask(self, inputs, mask=None):
        # Preserve incoming mask
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model
        })
        return config


class AttentionWeightsLayer(layers.Layer):
    """
    Custom layer that applies multi-head attention and extracts aggregated attention weights.

    Similar to AttentionLSTM but for transformer architecture.
    Supports masking for padded timesteps - masked positions get zero attention weight.
    """

    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        super(AttentionWeightsLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.supports_masking = True

    def build(self, input_shape):
        # Dense layer to compute attention scores
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )

        # MultiHeadAttention for computing the actual attention output
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )
        super(AttentionWeightsLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Apply multi-head attention
        attention_output = self.mha(inputs, inputs)

        # Compute attention weights (simplified scoring like in AttentionLSTM)
        # This gives us interpretable weights for visualization
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, seq_len, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, seq_len)

        # Apply mask: set attention scores for padded timesteps to -inf
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            e = tf.where(mask, e, tf.ones_like(e) * -1e9)

        # Compute attention weights using softmax (sum to 1.0)
        attention_weights = tf.nn.softmax(e, axis=1)  # (batch, seq_len)

        return attention_output, attention_weights

    def compute_mask(self, inputs, mask=None):
        # Don't propagate mask further (attention output doesn't need it)
        return mask

    def get_config(self):
        config = super(AttentionWeightsLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config


class TransformerSequenceModel:
    """
    Transformer-based model for sequence value prediction with masking support.

    Features:
    - Masking layer to handle padded sequences (mask_value=0.0)
    - Multi-head self-attention mechanism
    - Positional encoding
    - Returns both value prediction and attention weights
    """

    def __init__(self, input_shape: tuple, num_heads: int = 4, d_model: int = 128, ff_dim: int = 512, num_blocks: int = 2, dropout: float = 0.1):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.model = None

    def transformer_encoder(self, inputs, is_last_block=False):
        """
        Transformer encoder block with multi-head attention and feed-forward network.

        Args:
            inputs: Input tensor
            is_last_block: If True, returns attention weights from this block
        """
        if is_last_block:
            # Last block: capture attention weights
            attention_layer = AttentionWeightsLayer(
                num_heads=self.num_heads,
                key_dim=self.d_model,
                name='attention_weights_layer'
            )
            attention_output, attention_weights = attention_layer(inputs)
        else:
            # Regular block: just attention output
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model
            )(inputs, inputs)
            attention_weights = None

        attention_output = layers.Dropout(self.dropout)(attention_output)
        # Use layers.Add to preserve masking
        attention_output = layers.Add()([inputs, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # Feed-forward network
        ff_output = layers.Dense(self.ff_dim, activation='relu')(attention_output)
        ff_output = layers.Dense(self.d_model)(ff_output)
        ff_output = layers.Dropout(self.dropout)(ff_output)
        # Use layers.Add to preserve masking
        output = layers.Add()([attention_output, ff_output])
        output = layers.LayerNormalization(epsilon=1e-6)(output)

        if is_last_block:
            return output, attention_weights
        return output

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')

        # Masking layer: ignore padded timesteps (all zeros)
        x = layers.Masking(mask_value=0.0)(inputs)

        # Project to d_model dimensions
        x = layers.Dense(self.d_model)(x)

        # Add positional encoding (preserves mask)
        x = AddPositionalEncoding(
            max_seq_len=self.input_shape[0],
            d_model=self.d_model
        )(x)

        # Stack transformer blocks (all but last)
        for i in range(self.num_blocks - 1):
            x = self.transformer_encoder(x, is_last_block=False)

        # Last transformer block: capture attention weights
        x, attention_weights = self.transformer_encoder(x, is_last_block=True)

        # Aggregate sequence (mean pooling with masking support)
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        value_output = layers.Dense(1, activation='linear', name='value')(x)

        # Create model with outputs: value and attention weights
        self.model = keras.Model(
            inputs=inputs,
            outputs={'value': value_output, 'attention_weights': attention_weights}
        )
        return self.model

    def compile(self, learning_rate: float = 0.001):
        """Compile model with MSE loss for value and dummy loss for attention weights."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'value': 'mse',
                'attention_weights': 'mse'  # Dummy loss, weight is 0
            },
            loss_weights={
                'value': 1.0,
                'attention_weights': 0.0  # No contribution to total loss
            },
            metrics={'value': ['mae']}
        )
