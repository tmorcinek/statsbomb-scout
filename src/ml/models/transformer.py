import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AddPositionalEncoding(layers.Layer):

    def __init__(self, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.supports_masking = True

    def build(self, input_shape):
        self.position_embedding = layers.Embedding(
            input_dim=self.max_seq_len,
            output_dim=self.d_model,
            name='position_embedding'
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model
        })
        return config


class MultiHeadAttentionWithWeights(layers.Layer):

    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attn_mask = None
        if mask is not None:
            attn_mask = tf.cast(mask[:, tf.newaxis, :], tf.int32)

        attention_output, attention_scores = self.mha(
            inputs, inputs,
            attention_mask=attn_mask,
            return_attention_scores=True
        )

        attention_weights = tf.reduce_mean(attention_scores, axis=[1, 2])
        if mask is not None:
            mask_float = tf.cast(mask, dtype=attention_weights.dtype)
            attention_weights = attention_weights * mask_float
            sum_weights = tf.reduce_sum(attention_weights, axis=1, keepdims=True)
            sum_weights = tf.maximum(sum_weights, 1e-9)
            attention_weights = attention_weights / sum_weights

        return attention_output, attention_weights

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config


class TransformerSequenceModel:

    def __init__(self, input_shape: tuple, num_heads: int = 4, d_model: int = 128, ff_dim: int = 256, num_blocks: int = 2, dropout: float = 0.1):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.model = None

    def transformer_encoder(self, inputs):
        key_dim = self.d_model // self.num_heads
        attention_layer = MultiHeadAttentionWithWeights(
            num_heads=self.num_heads,
            key_dim=key_dim
        )
        attention_output, attention_weights = attention_layer(inputs)
        attention_output = layers.Dropout(self.dropout)(attention_output)
        attention_output = layers.Add()([inputs, attention_output])
        attention_output = layers.LayerNormalization()(attention_output)
        ff_output = layers.Dense(self.ff_dim, activation='relu')(attention_output)
        ff_output = layers.Dense(self.d_model)(ff_output)
        ff_output = layers.Dropout(self.dropout)(ff_output)
        output = layers.Add()([attention_output, ff_output])
        output = layers.LayerNormalization()(output)
        return output, attention_weights

    def build(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')
        x = layers.Masking(mask_value=0.0)(inputs)
        x = layers.Dense(self.d_model)(x)
        x = AddPositionalEncoding(
            max_seq_len=self.input_shape[0],
            d_model=self.d_model
        )(x)
        for _ in range(self.num_blocks - 1):
            x, _ = self.transformer_encoder(x)
        x, attention_weights = self.transformer_encoder(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        value_output = layers.Dense(1, activation='linear', name='value')(x)
        attention_output = layers.Lambda(lambda x: x, name='attention_weights')(attention_weights)
        self.model = keras.Model(
            inputs=inputs,
            outputs={'value': value_output, 'attention_weights': attention_output}
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
