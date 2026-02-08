import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):

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
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            e = tf.where(mask, e, tf.ones_like(e) * -1e9)
        attention_weights = tf.nn.softmax(e, axis=1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
        context = tf.reduce_sum(x * attention_weights_expanded, axis=1)
        return context, attention_weights

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class AttentionLSTMModel:

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')
        x = layers.Masking(mask_value=0.0)(inputs)
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(x)
        x = layers.Dropout(self.dropout)(x)
        lstm_out = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        lstm_out = layers.Dropout(self.dropout)(lstm_out)
        context, attention_weights = AttentionLayer(name='attention_weights')(lstm_out)
        x = layers.Dense(64, activation='relu')(context)
        x = layers.Dropout(self.dropout)(x)
        value_output = layers.Dense(1, activation='linear', name='value')(x)
        self.model = keras.Model(
            inputs=inputs,
            outputs={'value': value_output, 'attention_weights': attention_weights}
        )

        return self.model

    def compile(self, learning_rate: float = 0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'value': 'mse',
                'attention_weights': 'mse'
            },
            loss_weights={
                'value': 1.0,
                'attention_weights': 0.0
            },
            metrics={'value': ['mae']}
        )
