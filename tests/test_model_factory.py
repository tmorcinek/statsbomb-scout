import pytest
from tensorflow import keras

from src.ml.models.model_factory import create_model


def test_create_model_lstm():
    model = create_model('lstm', (10, 20))
    assert isinstance(model, keras.Model)


def test_create_model_transformer():
    model = create_model('transformer', (10, 20))
    assert isinstance(model, keras.Model)


def test_create_model_attention_lstm():
    model = create_model('attention_lstm', (10, 20))
    assert isinstance(model, keras.Model)


def test_create_model_unknown_type():
    with pytest.raises(ValueError, match="Unknown model type"):
        create_model('unknown', (10, 20))
