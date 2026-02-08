import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    """
    Custom Attention layer that computes importance weights for each timestep.

    The attention mechanism learns which actions in the sequence are most important
    for predicting the final value. Weights sum to 1.0 across the sequence.

    Supports masking: padded timesteps (masked) receive zero attention weight.
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
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
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # Compute attention scores: e_t = tanh(h_t * W + b)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, seq_len, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, seq_len)

        # Apply mask: set attention scores for padded timesteps to -inf
        # Dzięki temu po softmax padded timesteps mają wagę 0.0
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)  # (batch, seq_len)
            e = tf.where(mask, e, tf.ones_like(e) * -1e9)

        # Compute attention weights using softmax (sum to 1.0)
        attention_weights = tf.nn.softmax(e, axis=1)  # (batch, seq_len)

        # Compute weighted context vector
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)  # (batch, seq_len, 1)
        context = tf.reduce_sum(x * attention_weights_expanded, axis=1)  # (batch, hidden_dim)

        return context, attention_weights

    def compute_mask(self, inputs, mask=None):
        # Don't propagate mask further (context vector has no time dimension)
        return None

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class AttentionLSTMModel:
    """
    LSTM model with attention mechanism for sequence value prediction.
    """

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')

        # Masking layer: ignoruje kroki paddingowane zerami (0.0)
        # Dzięki temu LSTM nie uwzględnia padding'u w obliczeniach gradientów i stanów
        x = layers.Masking(mask_value=0.0)(inputs)

        # Bidirectional LSTM to capture context from both directions
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(x)
        x = layers.Dropout(self.dropout)(x)

        # Second LSTM layer (also return sequences for attention)
        lstm_out = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        lstm_out = layers.Dropout(self.dropout)(lstm_out)

        # Apply attention mechanism
        context, attention_weights = AttentionLayer(name='attention_weights')(lstm_out)

        # Dense layers for value prediction
        x = layers.Dense(64, activation='relu')(context)
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
