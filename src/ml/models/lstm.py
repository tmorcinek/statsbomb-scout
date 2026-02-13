from tensorflow import keras
from tensorflow.keras import layers


class LSTMSequenceModel:

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2, dense_units: int = 64, use_second_lstm: bool = True):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.use_second_lstm = use_second_lstm
        self.dense_units = dense_units
        self.model = None

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Masking(mask_value=0.0)(inputs)

        if self.use_second_lstm:
            x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.Dropout(self.dropout)(x)
            x = layers.LSTM(self.lstm_units // 2)(x)
            x = layers.Dropout(self.dropout)(x)
        else:
            x = layers.LSTM(self.lstm_units)(x)
        if self.dense_units > 1:
            x = layers.Dense(self.dense_units, activation='relu')(x)
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
