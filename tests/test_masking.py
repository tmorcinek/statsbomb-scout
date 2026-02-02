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
        from src.ml.models.transformer import TransformerSequenceModel

        sequence_length = 10
        num_features = 5
        batch_size = 2

        # Pierwsza sekwencja: 7 prawdziwych timesteps + 3 padded (0.0)
        # Druga sekwencja: 4 prawdziwe timesteps + 6 padded (0.0)
        X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
        X[0, 7:, :] = 0.0  # Padding dla pierwszej sekwencji
        X[1, 4:, :] = 0.0  # Padding dla drugiej sekwencji

        return X, sequence_length, num_features

    def test_transformer_model_has_masking_layer(self, test_data):
        """Test czy model Transformer zawiera warstwę Masking."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()

        # Sprawdź czy druga warstwa to Masking (pierwsza to Input)
        assert len(model.layers) >= 2, "Model powinien mieć co najmniej 2 warstwy"
        masking_layer = model.layers[1]
        assert isinstance(masking_layer, layers.Masking), f"Druga warstwa powinna być Masking, a jest {type(masking_layer)}"

    def test_transformer_model_has_attention_weights_output(self, test_data):
        """Test czy model Transformer zwraca attention weights."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()
        transformer_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)

        # Sprawdź czy outputs jest dict z 'value' i 'attention_weights'
        assert isinstance(outputs, dict), "Outputs powinny być dict"
        assert 'value' in outputs, "Outputs powinny zawierać 'value'"
        assert 'attention_weights' in outputs, "Outputs powinny zawierać 'attention_weights'"

    def test_transformer_attention_weights_shape(self, test_data):
        """Test czy attention weights mają poprawny kształt."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()
        transformer_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sprawdź kształt attention weights
        expected_shape = (X.shape[0], sequence_length)
        assert attention_weights.shape == expected_shape, \
            f"Attention weights powinny mieć kształt {expected_shape}, a mają {attention_weights.shape}"

    def test_transformer_attention_weights_sum_to_one(self, test_data):
        """Test czy wagi uwagi sumują się do 1.0."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()
        transformer_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sprawdź czy wagi sumują się do ~1.0 dla każdej sekwencji
        for i in range(attention_weights.shape[0]):
            weight_sum = attention_weights[i].sum()
            assert np.isclose(weight_sum, 1.0, atol=1e-5), \
                f"Wagi uwagi dla sekwencji {i} powinny sumować się do 1.0, a sumują się do {weight_sum}"

    def test_transformer_attention_weights_zero_for_padding(self, test_data):
        """Test czy wagi uwagi dla paddingowanych kroków są zerowe."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()
        transformer_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
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

    def test_transformer_attention_weights_non_zero_for_valid_steps(self, test_data):
        """Test czy wagi uwagi dla prawdziwych kroków są większe od zera."""
        from src.ml.models.transformer import TransformerSequenceModel

        X, sequence_length, num_features = test_data

        transformer_model = TransformerSequenceModel(
            input_shape=(sequence_length, num_features),
            num_heads=2,
            d_model=32,
            ff_dim=64,
            num_blocks=2,
            dropout=0.2
        )
        model = transformer_model.build()
        transformer_model.compile(learning_rate=0.001)

        outputs = model.predict(X, verbose=0)
        attention_weights = outputs['attention_weights']

        # Sekwencja 1: prawdziwe kroki 0-6
        valid_weights_seq1 = attention_weights[0, :7]
        assert np.all(valid_weights_seq1 > 0), \
            f"Wszystkie wagi dla prawdziwych kroków sekwencji 1 powinny być > 0"

        # Sekwencja 2: prawdziwe kroki 0-3
        valid_weights_seq2 = attention_weights[1, :4]
        assert np.all(valid_weights_seq2 > 0), \
            f"Wszystkie wagi dla prawdziwych kroków sekwencji 2 powinny być > 0"

