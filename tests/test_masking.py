import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from src.ml.models.attention_lstm import AttentionLSTMModel
from src.ml.models.lstm import LSTMSequenceModel

tf.keras.utils.set_random_seed(123)
np.random.seed(123)


class TestLSTMModelMasking:

    @pytest.fixture
    def test_data(self):
        """Fixture z danymi testowymi: sekwencje z paddingiem."""
        sequence_length = 10
        num_features = 5
        batch_size = 2

        # Pierwsza sekwencja: 7 prawdziwych timesteps + 3 padded (0.0)
        # Druga sekwencja: 4 prawdziwe timesteps + 6 padded (0.0)
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0  # Padding dla pierwszej sekwencji
        X[1, 4:, :] = 0.0  # Padding dla drugiej sekwencji

        return X, sequence_length, num_features

    def test_lstm_model_has_masking_layer(self, test_data):
        """Test czy model LSTM zawiera warstwę Masking."""
        X, sequence_length, num_features = test_data

        lstm_model = LSTMSequenceModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = lstm_model.build()

        # Sprawdź czy druga warstwa to Masking (pierwsza to Input)
        assert len(model.layers) >= 2, "Model powinien mieć co najmniej 2 warstwy"
        masking_layer = model.layers[1]
        assert isinstance(masking_layer, layers.Masking), f"Druga warstwa powinna być Masking, a jest {type(masking_layer)}"

    def test_lstm_masking_layer_has_correct_mask_value(self, test_data):
        """Test czy warstwa Masking ma poprawną wartość mask_value=0.0."""
        X, sequence_length, num_features = test_data

        lstm_model = LSTMSequenceModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = lstm_model.build()

        masking_layer = model.layers[1]
        assert masking_layer.mask_value == 0.0, f"mask_value powinno być 0.0, a jest {masking_layer.mask_value}"

    def test_lstm_model_compiles_successfully(self, test_data):
        """Test czy model LSTM kompiluje się poprawnie z maskowaniem."""
        X, sequence_length, num_features = test_data

        lstm_model = LSTMSequenceModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = lstm_model.build()
        lstm_model.compile(learning_rate=0.001)

        # Sprawdź czy model został skompilowany
        assert model.optimizer is not None, "Model powinien mieć optimizer po kompilacji"
        assert model.loss is not None, "Model powinien mieć loss po kompilacji"

    def test_lstm_model_makes_predictions_with_padding(self, test_data):
        """Test czy model LSTM wykonuje predykcje dla danych z paddingiem."""
        X, sequence_length, num_features = test_data

        lstm_model = LSTMSequenceModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = lstm_model.build()
        lstm_model.compile(learning_rate=0.001)

        predictions = model.predict(X, verbose=0)

        # Sprawdź kształt predykcji
        assert predictions.shape == (2, 1), f"Predykcje powinny mieć kształt (2, 1), a mają {predictions.shape}"

        # Sprawdź czy predykcje są liczbami (nie NaN ani inf)
        assert not np.isnan(predictions).any(), "Predykcje zawierają NaN"
        assert not np.isinf(predictions).any(), "Predykcje zawierają inf"

    def test_lstm_model_handles_different_padding_lengths(self):
        """Test czy model LSTM radzi sobie z różnymi długościami paddingu."""
        sequence_length = 15
        num_features = 8
        batch_size = 3

        # Trzy sekwencje z różnymi długościami paddingu
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 10:, :] = 0.0  # 10 prawdziwych, 5 paddingowanych
        X[1, 5:, :] = 0.0  # 5 prawdziwych, 10 paddingowanych
        X[2, 12:, :] = 0.0  # 12 prawdziwych, 3 paddingowane

        lstm_model = LSTMSequenceModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = lstm_model.build()
        lstm_model.compile(learning_rate=0.001)

        predictions = model.predict(X, verbose=0)

        # Sprawdź czy wszystkie predykcje są poprawne
        assert predictions.shape == (3, 1), f"Predykcje powinny mieć kształt (3, 1), a mają {predictions.shape}"
        assert not np.isnan(predictions).any(), "Predykcje zawierają NaN"


class TestAttentionLSTMModelMasking:

    @pytest.fixture
    def test_data(self):
        """Fixture z danymi testowymi: sekwencje z paddingiem."""
        sequence_length = 10
        num_features = 5
        batch_size = 2

        # Pierwsza sekwencja: 7 prawdziwych timesteps + 3 padded (0.0)
        # Druga sekwencja: 4 prawdziwe timesteps + 6 padded (0.0)
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0  # Padding dla pierwszej sekwencji
        X[1, 4:, :] = 0.0  # Padding dla drugiej sekwencji

        return X, sequence_length, num_features

    def test_attention_lstm_model_has_masking_layer(self, test_data):
        """Test czy model Attention LSTM zawiera warstwę Masking."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()

        # Sprawdź czy druga warstwa to Masking (pierwsza to Input)
        assert len(model.layers) >= 2, "Model powinien mieć co najmniej 2 warstwy"
        masking_layer = model.layers[1]
        assert isinstance(masking_layer, layers.Masking), f"Druga warstwa powinna być Masking, a jest {type(masking_layer)}"

    def test_attention_lstm_masking_layer_has_correct_mask_value(self, test_data):
        """Test czy warstwa Masking ma poprawną wartość mask_value=0.0."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()

        masking_layer = model.layers[1]
        assert masking_layer.mask_value == 0.0, f"mask_value powinno być 0.0, a jest {masking_layer.mask_value}"

    def test_attention_lstm_model_compiles_successfully(self, test_data):
        """Test czy model Attention LSTM kompiluje się poprawnie z maskowaniem."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        # Sprawdź czy model został skompilowany
        assert model.optimizer is not None, "Model powinien mieć optimizer po kompilacji"

    def test_attention_lstm_model_makes_predictions_with_padding(self, test_data):
        """Test czy model Attention LSTM wykonuje predykcje dla danych z paddingiem."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        predictions = outputs['value']
        attention_weights = outputs['attention_weights']

        # Sprawdź kształt predykcji
        assert predictions.shape == (2, 1), f"Predykcje powinny mieć kształt (2, 1), a mają {predictions.shape}"

        # Sprawdź kształt wag uwagi
        assert attention_weights.shape == (2, sequence_length), f"Wagi uwagi powinny mieć kształt (2, {sequence_length}), a mają {attention_weights.shape}"

        # Sprawdź czy predykcje są liczbami (nie NaN ani inf)
        assert not np.isnan(predictions).any(), "Predykcje zawierają NaN"
        assert not np.isinf(predictions).any(), "Predykcje zawierają inf"

    def test_attention_weights_sum_to_one(self, test_data):
        """Test czy wagi uwagi sumują się do 1.0 dla każdej sekwencji."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sprawdź czy wagi sumują się do ~1.0 dla każdej sekwencji
        for i in range(attention_weights.shape[0]):
            weight_sum = attention_weights[i].sum()
            assert np.isclose(weight_sum, 1.0, atol=1e-5), f"Wagi uwagi dla sekwencji {i} powinny sumować się do 1.0, a sumują się do {weight_sum}"

    def test_attention_weights_zero_for_padding(self, test_data):
        """Test czy wagi uwagi dla paddingowanych kroków są bliskie zeru."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sekwencja 1: padding od indeksu 7
        # Wagi dla indeksów 7, 8, 9 powinny być ~0
        padding_weights_seq1 = attention_weights[0, 7:]
        assert np.allclose(padding_weights_seq1, 0.0, atol=1e-6), f"Wagi dla paddingu sekwencji 1 powinny być ~0, a są {padding_weights_seq1}"

        # Sekwencja 2: padding od indeksu 4
        # Wagi dla indeksów 4-9 powinny być ~0
        padding_weights_seq2 = attention_weights[1, 4:]
        assert np.allclose(padding_weights_seq2, 0.0, atol=1e-6), f"Wagi dla paddingu sekwencji 2 powinny być ~0, a są {padding_weights_seq2}"

    def test_attention_weights_non_zero_for_valid_steps(self, test_data):
        """Test czy wagi uwagi dla prawdziwych kroków są większe od zera."""
        X, sequence_length, num_features = test_data

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sekwencja 1: prawdziwe kroki 0-6
        valid_weights_seq1 = attention_weights[0, :7]
        assert np.all(valid_weights_seq1 > 0), f"Wszystkie wagi dla prawdziwych kroków sekwencji 1 powinny być > 0"

        # Sekwencja 2: prawdziwe kroki 0-3
        valid_weights_seq2 = attention_weights[1, :4]
        assert np.all(valid_weights_seq2 > 0), f"Wszystkie wagi dla prawdziwych kroków sekwencji 2 powinny być > 0"

    def test_attention_lstm_handles_different_padding_lengths(self):
        """Test czy model Attention LSTM radzi sobie z różnymi długościami paddingu."""
        sequence_length = 15
        num_features = 8
        batch_size = 3

        # Trzy sekwencje z różnymi długościami paddingu
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 10:, :] = 0.0  # 10 prawdziwych, 5 paddingowanych
        X[1, 5:, :] = 0.0  # 5 prawdziwych, 10 paddingowanych
        X[2, 12:, :] = 0.0  # 12 prawdziwych, 3 paddingowane

        attn_lstm_model = AttentionLSTMModel(
            input_shape=(sequence_length, num_features),
            lstm_units=32,
            dropout=0.2
        )
        model = attn_lstm_model.build()
        attn_lstm_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        predictions = outputs['value']
        attention_weights = outputs['attention_weights']

        # Sprawdź kształty
        assert predictions.shape == (3, 1), f"Predykcje powinny mieć kształt (3, 1), a mają {predictions.shape}"
        assert attention_weights.shape == (3, sequence_length), f"Wagi uwagi powinny mieć kształt (3, {sequence_length}), a mają {attention_weights.shape}"

        # Sprawdź czy padding ma wagę 0 dla każdej sekwencji
        assert np.allclose(attention_weights[0, 10:], 0.0, atol=1e-6), "Padding sekwencji 1 powinien mieć wagę ~0"
        assert np.allclose(attention_weights[1, 5:], 0.0, atol=1e-6), "Padding sekwencji 2 powinien mieć wagę ~0"
        assert np.allclose(attention_weights[2, 12:], 0.0, atol=1e-6), "Padding sekwencji 3 powinien mieć wagę ~0"


class TestTransformerModelMasking:

    @pytest.fixture
    def test_data(self):
        """Fixture z danymi testowymi: sekwencje z paddingiem."""
        sequence_length = 10
        num_features = 5
        batch_size = 2

        # Pierwsza sekwencja: 7 prawdziwych timesteps + 3 padded (0.0)
        # Druga sekwencja: 4 prawdziwe timesteps + 6 padded (0.0)
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0  # Padding dla pierwszej sekwencji
        X[1, 4:, :] = 0.0  # Padding dla drugiej sekwencji

        return X, sequence_length, num_features

    @pytest.fixture
    def transformer_model(self, test_data):
        """Fixture z gotowym modelem Transformer."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        model_builder = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = model_builder.build()
        model_builder.compile(learning_rate=0.001)
        return model

    def test_transformer_model_has_masking_layer(self, transformer_model):
        """Test czy model Transformer zawiera warstwę Masking."""
        # Sprawdź czy model ma warstwę Masking (szukaj w całym modelu, nie zakładaj pozycji)
        masking_layer = next((layer for layer in transformer_model.layers
                              if isinstance(layer, layers.Masking)), None)
        assert masking_layer is not None, "Model powinien zawierać warstwę Masking"
        assert masking_layer.mask_value == 0.0, f"mask_value powinno być 0.0, a jest {masking_layer.mask_value}"

    def test_transformer_model_has_attention_weights_output(self, test_data, transformer_model):
        """Test czy model Transformer zwraca attention weights."""
        X, sequence_length, num_features = test_data

        outputs = transformer_model.predict(X, verbose=0)

        # Sprawdź czy outputs jest dict z 'value' i 'attention_weights'
        assert isinstance(outputs, dict), "Outputs powinny być dict"
        assert 'value' in outputs, "Outputs powinny zawierać 'value'"
        assert 'attention_weights' in outputs, "Outputs powinny zawierać 'attention_weights'"

    def test_transformer_attention_weights_shape(self, test_data, transformer_model):
        """Test czy attention weights mają poprawny kształt."""
        X, sequence_length, num_features = test_data

        outputs = transformer_model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sprawdź kształt attention weights
        expected_shape = (X.shape[0], sequence_length)
        assert attention_weights.shape == expected_shape, \
            f"Attention weights powinny mieć kształt {expected_shape}, a mają {attention_weights.shape}"

    def test_transformer_attention_weights_sum_to_one(self, test_data, transformer_model):
        """Test czy wagi uwagi sumują się do 1.0."""
        X, sequence_length, num_features = test_data

        outputs = transformer_model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sprawdź czy wagi sumują się do ~1.0 dla każdej sekwencji
        for i in range(attention_weights.shape[0]):
            weight_sum = attention_weights[i].sum()
            assert np.isclose(weight_sum, 1.0, atol=1e-5), \
                f"Wagi uwagi dla sekwencji {i} powinny sumować się do 1.0, a sumują się do {weight_sum}"

    def test_transformer_attention_weights_zero_for_padding(self, test_data, transformer_model):
        """Test czy wagi uwagi dla paddingowanych kroków są zerowe."""
        X, sequence_length, num_features = test_data

        outputs = transformer_model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sekwencja 1: padding od indeksu 7
        # Wagi dla indeksów 7, 8, 9 powinny być ~0
        padding_weights_seq1 = attention_weights[0, 7:]
        assert np.allclose(padding_weights_seq1, 0.0, atol=1e-6), \
            f"Wagi dla paddingu sekwencji 1 powinny być ~0, a są {padding_weights_seq1}"

        # Sekwencja 2: padding od indeksu 4
        # Wagi dla indeksów 4-9 powinny być ~0
        padding_weights_seq2 = attention_weights[1, 4:]
        assert np.allclose(padding_weights_seq2, 0.0, atol=1e-6), \
            f"Wagi dla paddingu sekwencji 2 powinny być ~0, a są {padding_weights_seq2}"

    def test_transformer_attention_weights_non_zero_for_valid_steps(self, test_data, transformer_model):
        """Test czy wagi uwagi dla prawdziwych kroków są większe od zera."""
        X, sequence_length, num_features = test_data

        outputs = transformer_model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sekwencja 1: prawdziwe kroki 0-6
        valid_weights_seq1 = attention_weights[0, :7]
        assert np.all(valid_weights_seq1 > 0), \
            f"Wszystkie wagi dla prawdziwych kroków sekwencji 1 powinny być > 0"

        # Sekwencja 2: prawdziwe kroki 0-3
        valid_weights_seq2 = attention_weights[1, :4]
        assert np.all(valid_weights_seq2 > 0), \
            f"Wszystkie wagi dla prawdziwych kroków sekwencji 2 powinny być > 0"

class TestTransformerTrueAttentionModelMasking:

    @pytest.fixture
    def test_data(self):
        sequence_length = 10
        num_features = 5
        batch_size = 2

        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0
        X[1, 4:, :] = 0.0

        return X, sequence_length, num_features

    @pytest.fixture
    def transformer_true_attention_model(self, test_data):
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        model_builder = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2,
            true_attention=True
        )
        model = model_builder.build()
        model_builder.compile(learning_rate=0.001)
        return model

    def test_true_attention_outputs_dict(self, test_data, transformer_true_attention_model):
        X, *_ = test_data
        outputs = transformer_true_attention_model.predict(X, verbose=0)
        assert isinstance(outputs, dict)
        assert 'value' in outputs
        assert 'attention_weights' in outputs

    def test_true_attention_weights_shape(self, test_data, transformer_true_attention_model):
        X, sequence_length, _ = test_data
        attention_weights = transformer_true_attention_model.predict(X, verbose=0)['attention_weights']
        assert attention_weights.shape == (X.shape[0], sequence_length)

    def test_true_attention_weights_sum_to_one(self, test_data, transformer_true_attention_model):
        X, *_ = test_data
        attention_weights = transformer_true_attention_model.predict(X, verbose=0)['attention_weights']
        sums = np.sum(attention_weights, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_true_attention_weights_zero_for_padding(self, test_data, transformer_true_attention_model):
        X, *_ = test_data
        attention_weights = transformer_true_attention_model.predict(X, verbose=0)['attention_weights']
        assert np.allclose(attention_weights[0, 7:], 0.0, atol=1e-6)
        assert np.allclose(attention_weights[1, 4:], 0.0, atol=1e-6)

    def test_true_attention_weights_positive_for_valid_steps(self, test_data, transformer_true_attention_model):
        X, *_ = test_data
        attention_weights = transformer_true_attention_model.predict(X, verbose=0)['attention_weights']
        assert np.all(attention_weights[0, :7] > 0)
        assert np.all(attention_weights[1, :4] > 0)


class TestBiGRUModelMasking:
    """Tests for BiGRU model masking and attention weights."""

    @pytest.fixture
    def test_data(self):
        """Fixture with test data: sequences with padding."""
        sequence_length = 10
        num_features = 5
        batch_size = 2

        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0  # Padding for first sequence
        X[1, 4:, :] = 0.0  # Padding for second sequence

        return X, sequence_length, num_features

    @pytest.fixture
    def bigru_model(self, test_data):
        """Fixture with compiled BiGRU model."""
        from src.ml.models.bigru import build_seq_value_model

        _, sequence_length, num_features = test_data
        model = build_seq_value_model(
            input_shape=(sequence_length, num_features),
            rnn_units=32,
            attn_hidden=16,
            dropout=0.2,
            return_attention=True  # W testach chcemy attention
        )
        model.compile(
            optimizer='adam',
            loss={'value': 'mse', 'attention_weights': 'mse'},
            loss_weights={'value': 1.0, 'attention_weights': 0.0},
            metrics={'value': ['mae']}
        )
        return model

    def test_bigru_has_masking_layer(self, test_data):
        """Test if BiGRU model has Masking layer."""
        from src.ml.models.bigru import build_seq_value_model

        _, sequence_length, num_features = test_data
        model = build_seq_value_model(
            input_shape=(sequence_length, num_features),
            rnn_units=32,
            attn_hidden=16,
            dropout=0.2,
            return_attention=False
        )

        # Second layer should be Masking (first is Input)
        assert len(model.layers) >= 2
        masking_layer = model.layers[1]
        assert isinstance(masking_layer, layers.Masking)
        assert masking_layer.mask_value == 0.0

    def test_bigru_outputs_dict(self, test_data, bigru_model):
        """Test if BiGRU outputs dictionary with 'value' and 'attention_weights'."""
        X, *_ = test_data
        outputs = bigru_model.predict(X, verbose=0)

        assert isinstance(outputs, dict)
        assert 'value' in outputs
        assert 'attention_weights' in outputs

    def test_bigru_attention_weights_shape(self, test_data, bigru_model):
        """Test if attention weights have correct shape (batch, seq_len)."""
        X, sequence_length, _ = test_data
        outputs = bigru_model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        assert attention_weights.shape == (2, sequence_length)

    def test_bigru_attention_weights_sum_to_one(self, test_data, bigru_model):
        """Test if attention weights sum to 1.0 for each sequence."""
        X, *_ = test_data
        attention_weights = bigru_model.predict(X, verbose=0)['attention_weights']

        for i in range(attention_weights.shape[0]):
            weight_sum = attention_weights[i].sum()
            assert np.isclose(weight_sum, 1.0, atol=1e-5), \
                f"Attention weights for sequence {i} should sum to 1.0, got {weight_sum}"

    def test_bigru_attention_weights_zero_for_padding(self, test_data, bigru_model):
        """Test if attention weights are zero for padded timesteps."""
        X, *_ = test_data
        attention_weights = bigru_model.predict(X, verbose=0)['attention_weights']

        # Sequence 0: padding from index 7
        assert np.allclose(attention_weights[0, 7:], 0.0, atol=1e-6), \
            f"Padding weights for seq 0 should be ~0, got {attention_weights[0, 7:]}"

        # Sequence 1: padding from index 4
        assert np.allclose(attention_weights[1, 4:], 0.0, atol=1e-6), \
            f"Padding weights for seq 1 should be ~0, got {attention_weights[1, 4:]}"

    def test_bigru_attention_weights_positive_for_valid_steps(self, test_data, bigru_model):
        """Test if attention weights are positive for valid (non-padded) timesteps."""
        X, *_ = test_data
        attention_weights = bigru_model.predict(X, verbose=0)['attention_weights']

        # Sequence 0: valid timesteps 0-6
        assert np.all(attention_weights[0, :7] > 0), \
            f"Valid timesteps should have positive weights, got {attention_weights[0, :7]}"

        # Sequence 1: valid timesteps 0-3
        assert np.all(attention_weights[1, :4] > 0), \
            f"Valid timesteps should have positive weights, got {attention_weights[1, :4]}"

    def test_bigru_handles_different_padding_lengths(self, bigru_model):
        """Test if BiGRU handles sequences with different amounts of padding."""
        # Create sequences with varying padding
        X1 = np.random.randn(1, 10, 5).astype(np.float32)
        X1[0, 2:, :] = 0.0  # Only 2 valid timesteps

        X2 = np.random.randn(1, 10, 5).astype(np.float32)
        X2[0, 9:, :] = 0.0  # 9 valid timesteps

        outputs1 = bigru_model.predict(X1, verbose=0)
        outputs2 = bigru_model.predict(X2, verbose=0)

        attn1 = outputs1['attention_weights'][0]
        attn2 = outputs2['attention_weights'][0]

        # Check masking works for both
        assert np.allclose(attn1[2:], 0.0, atol=1e-6)
        assert np.allclose(attn2[9:], 0.0, atol=1e-6)

        # Check valid steps have positive weights
        assert np.all(attn1[:2] > 0)
        assert np.all(attn2[:9] > 0)
