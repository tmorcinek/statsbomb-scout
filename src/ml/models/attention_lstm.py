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
    """

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2, return_attention: bool = True):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.return_attention = return_attention
        self.model = None

    def build(self) -> keras.Model:
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
    model_builder = AttentionLSTMModel(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout=lstm_dropout,
        return_attention=return_attention
    )

    model = model_builder.build()
    model_builder.compile(learning_rate=learning_rate)

    return model
