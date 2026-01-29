import numpy as np
import tensorflow as tf

from src.ml.models.attention_lstm import AttentionLSTMModel
from src.ml.models.lstm import LSTMSequenceModel

tf.keras.utils.set_random_seed(123)
np.random.seed(123)


def test_lstm_padding_invariance():
    """
    Test inwariancji na padding dla LSTM.

    Sprawdza czy predykcja sekwencji długości 7 jest taka sama
    jak predykcja tej samej sekwencji z paddingiem do długości 10.
    Dzięki maskowaniu wyniki powinny być niemal identyczne.
    """
    original_length = 7
    padded_length = 10
    num_features = 5

    # Utwórz sekwencję oryginalną (długość 7)
    X_original = np.random.randn(1, original_length, num_features).astype(np.float32)

    # Utwórz sekwencję z paddingiem (długość 10)
    X_padded = np.zeros((1, padded_length, num_features), dtype=np.float32)
    X_padded[0, :original_length, :] = X_original[0]  # Skopiuj oryginalne dane
    X_padded[0, original_length:, :] = 0.0  # Padding

    # Zbuduj model dla oryginalnej długości
    lstm_model_original = LSTMSequenceModel(
        input_shape=(original_length, num_features),
        lstm_units=32,
        dropout=0.0  # Bez dropout dla deterministycznego wyniku
    )
    model_original = lstm_model_original.build()
    lstm_model_original.compile(learning_rate=0.001)

    # Zbuduj model dla paddingowanej długości
    lstm_model_padded = LSTMSequenceModel(
        input_shape=(padded_length, num_features),
        lstm_units=32,
        dropout=0.0  # Bez dropout dla deterministycznego wyniku
    )
    model_padded = lstm_model_padded.build()
    lstm_model_padded.compile(learning_rate=0.001)

    # Skopiuj wagi z modelu oryginalnego do modelu z paddingiem
    # (aby oba modele miały te same parametry)
    for layer_orig, layer_padded in zip(model_original.layers, model_padded.layers):
        if layer_orig.get_weights():
            layer_padded.set_weights(layer_orig.get_weights())

    # Predykcje
    pred_original = model_original.predict(X_original, verbose=0)
    pred_padded = model_padded.predict(X_padded, verbose=0)

    # Sprawdź czy predykcje są niemal identyczne
    assert np.allclose(pred_original, pred_padded, rtol=1e-5, atol=1e-5), \
        f"Predykcje powinny być niemal identyczne:\n" \
        f"Oryginalna (len={original_length}): {pred_original[0, 0]}\n" \
        f"Z paddingiem (len={padded_length}): {pred_padded[0, 0]}\n" \
        f"Różnica: {abs(pred_original[0, 0] - pred_padded[0, 0])}"


def test_attention_lstm_padding_invariance():
    """
    Test inwariancji na padding dla Attention LSTM.

    Sprawdza czy przy tych samych wagach LSTM i Dense,
    model z paddingiem daje poprawne wyniki:
    - wagi uwagi dla paddingu są ~0
    - wagi uwagi dla prawdziwych kroków są > 0
    - suma wag = 1.0

    Uwaga: Nie porównujemy bezpośrednio predykcji między modelami
    o różnej długości sekwencji, bo AttentionLayer ma wagi zależne
    od seq_len (bias shape=(seq_len, 1)).
    """
    sequence_length = 10
    num_features = 5
    original_length = 7  # Pierwszych 7 kroków to prawdziwe dane

    # Utwórz sekwencję z paddingiem
    X = np.random.randn(1, sequence_length, num_features).astype(np.float32)
    X[0, original_length:, :] = 0.0  # Padding od indeksu 7

    # Zbuduj model
    attn_lstm_model = AttentionLSTMModel(
        input_shape=(sequence_length, num_features),
        lstm_units=32,
        dropout=0.0  # Bez dropout dla deterministycznego wyniku
    )
    model = attn_lstm_model.build()
    attn_lstm_model.compile(learning_rate=0.001)

    # Predykcje
    outputs = model.predict(X, verbose=0)
    predictions = outputs['value']
    attention_weights = outputs['attention_weights']

    # Test 1: Wagi uwagi dla paddingu powinny być ~0
    padding_weights = attention_weights[0, original_length:]
    assert np.allclose(padding_weights, 0.0, atol=1e-6), \
        f"Wagi uwagi dla paddingu (indeksy {original_length}-{sequence_length - 1}) " \
        f"powinny być ~0, a są {padding_weights}"

    # Test 2: Wagi uwagi dla prawdziwych kroków powinny być > 0
    valid_weights = attention_weights[0, :original_length]
    assert np.all(valid_weights > 0), \
        f"Wszystkie wagi uwagi dla prawdziwych kroków (indeksy 0-{original_length - 1}) " \
        f"powinny być > 0, a są {valid_weights}"

    # Test 3: Suma wag uwagi = 1.0
    weight_sum = attention_weights[0].sum()
    assert np.isclose(weight_sum, 1.0, atol=1e-5), \
        f"Suma wag uwagi powinna być 1.0, a jest {weight_sum}"

    # Test 4: Predykcja nie zawiera NaN ani inf
    assert not np.isnan(predictions).any(), "Predykcja zawiera NaN"
    assert not np.isinf(predictions).any(), "Predykcja zawiera inf"
