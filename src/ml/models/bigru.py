import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TemporalAttentionPooling(layers.Layer):
    """
    Temporal attention pooling layer with masking support.

    Produces:
      - context vector: sum_t a_t * x_t
      - attention weights a: shape (batch, seq_len), sum(a)=1 for each sample

    Works with Keras Masking (mask shape: (batch, seq_len)).
    Padded timesteps get zero attention weight.
    """

    def __init__(self, attn_hidden: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.attn_hidden = attn_hidden
        self.supports_masking = True

        self.score_net = tf.keras.Sequential([
            layers.Dense(attn_hidden, activation="tanh",
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(1, activation=None)
        ])

    def call(self, x, mask=None):
        # x: (batch, seq_len, d)
        logits = self.score_net(x)  # (batch, seq_len, 1)
        logits = tf.squeeze(logits, axis=-1)  # (batch, seq_len)

        if mask is not None:
            # mask: True for real tokens, False for padding
            mask = tf.cast(mask, logits.dtype)
            logits = logits + (1.0 - mask) * (-1e9)

        a = tf.nn.softmax(logits, axis=1)  # (batch, seq_len), sums to 1

        # context = sum_t a_t * x_t
        context = tf.einsum("bs,bsd->bd", a, x)  # (batch, d)
        return context, a

    def compute_mask(self, inputs, mask=None):
        # don't propagate mask further (context has no time dimension)
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attn_hidden": self.attn_hidden})
        return cfg


def build_seq_value_model(
    input_shape,
    rnn_units=128,
    attn_hidden=64,
    dropout=0.2,
    recurrent_dropout=0.15,
    l2_reg=0.01,
    return_attention=False
):
    """
    Build BiGRU model with temporal attention.

    Args:
        rnn_units: Number of GRU units
        attn_hidden: Attention hidden dimension
        dropout: Dropout rate for Dense layers
        recurrent_dropout: Dropout for recurrent connections (prevents overfitting in GRU)
        l2_reg: L2 regularization factor for Dense layers
        return_attention: If True, returns dict with value + attention_weights (for loading/analysis)
                         If False, returns only value (for training)
    """
    inp = layers.Input(shape=input_shape, name="sequence_input")
    x = layers.Masking(mask_value=0.0)(inp)

    x = layers.Bidirectional(
        layers.GRU(
            rnn_units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_constraint=keras.constraints.MaxNorm(3.0)
        ),
        name="bigru"
    )(x)
    x = layers.LayerNormalization()(x)

    context, attn = TemporalAttentionPooling(attn_hidden=attn_hidden, name="attn_pool")(x)

    h = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(context)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout)(h)

    value = layers.Dense(1, activation="linear", name="value")(h)

    if return_attention:
        attn_output = layers.Lambda(lambda x: x, name="attention_weights")(attn)
        model = keras.Model(
            inputs=inp,
            outputs={"value": value, "attention_weights": attn_output},
            name="bigru_model"
        )
    else:
        model = keras.Model(inputs=inp, outputs=value, name="bigru_model")

    return model

