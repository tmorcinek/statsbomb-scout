"""Attention-based LSTM model for sequence value prediction with action importance weights."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    """
    Custom Attention layer that computes importance weights for each timestep.

    The attention mechanism learns which actions in the sequence are most important
    for predicting the final value. Weights sum to 1.0 across the sequence.
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build attention layer weights.

        Args:
            input_shape: (batch_size, sequence_length, hidden_dim)
        """
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

    def call(self, x):
        """
        Compute attention weights and weighted context.

        Args:
            x: LSTM hidden states (batch_size, sequence_length, hidden_dim)

        Returns:
            context: Weighted sum of hidden states (batch_size, hidden_dim)
            attention_weights: Importance weights for each timestep (batch_size, sequence_length)
        """
        # Compute attention scores: e_t = tanh(h_t * W + b)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, seq_len, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, seq_len)

        # Compute attention weights using softmax (sum to 1.0)
        attention_weights = tf.nn.softmax(e, axis=1)  # (batch, seq_len)

        # Compute weighted context vector
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)  # (batch, seq_len, 1)
        context = tf.reduce_sum(x * attention_weights_expanded, axis=1)  # (batch, hidden_dim)

        return context, attention_weights

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class AttentionLSTMModel:
    """
    LSTM model with attention mechanism for sequence value prediction.

    This model:
    1. Processes sequence through bidirectional LSTM
    2. Computes attention weights (importance of each action)
    3. Returns both:
       - Predicted value (single float)
       - Attention weights (sequence_length floats summing to 1.0)
    """

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2, return_attention: bool = True):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.return_attention = return_attention
        self.model = None

    def build(self) -> keras.Model:
        """
        Build Attention LSTM model architecture.

        Returns:
            Compiled Keras model with two outputs: value and attention weights
        """
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')

        # Bidirectional LSTM to capture context from both directions
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(inputs)
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

        # Create model with two outputs
        if self.return_attention:
            self.model = keras.Model(
                inputs=inputs,
                outputs={'value': value_output, 'attention_weights': attention_weights}
            )
        else:
            self.model = keras.Model(inputs=inputs, outputs=value_output)

        return self.model

    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model.

        Only the value output has a loss - attention weights are learned implicitly
        through backpropagation to minimize value prediction error.

        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.return_attention:
            # Use MSE for value, dummy loss for attention_weights with 0 weight
            # This ensures metrics are properly tracked
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
        else:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )


def create_attention_lstm_model(input_shape: tuple, lstm_units: int = 128,
                                lstm_dropout: float = 0.2,
                                learning_rate: float = 0.001,
                                return_attention: bool = True) -> keras.Model:
    """
    Factory function to create and compile Attention LSTM model.

    Args:
        input_shape: Shape of input (sequence_length, n_features)
        lstm_units: Number of LSTM units
        lstm_dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        return_attention: If True, model returns attention weights

    Returns:
        Compiled Keras model
    """
    model_builder = AttentionLSTMModel(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout=lstm_dropout,
        return_attention=return_attention
    )

    model = model_builder.build()
    model_builder.compile(learning_rate=learning_rate)

    return model
