from tensorflow import keras
from tensorflow.keras import layers


class LSTMSequenceModel:
    """LSTM-based model for sequence value prediction."""

    def __init__(self, input_shape: tuple, lstm_units: int = 128, dropout: float = 0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape)

        # Masking layer: ignoruje kroki paddingowane zerami (0.0)
        # Dzięki temu LSTM nie uwzględnia padding'u w obliczeniach gradientów i stanów
        x = layers.Masking(mask_value=0.0)(inputs)

        # LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
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
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
