import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TemporalAttentionPooling(layers.Layer):

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
        logits = self.score_net(x)
        logits = tf.squeeze(logits, axis=-1)
        if mask is not None:
            mask = tf.cast(mask, logits.dtype)
            logits = logits + (1.0 - mask) * (-1e9)
        a = tf.nn.softmax(logits, axis=1)
        context = tf.einsum("bs,bsd->bd", a, x)
        return context, a

    def compute_mask(self, inputs, mask=None):
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

