import tensorflow as tf
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, Dense, LSTM, Bidirectional, Activation


def bahdanau_attention(values, units):
    """ Simplified Bahdanau Attention

    The query is omitted since we don't generate results auto-regressively.
    key = value = x, shape  (B, T, D)
    """

    x = Dense(units)(values)  # (B, T, units)
    x = tf.nn.tanh(x)
    score = Dense(1)(x)       # (B, T, 1)

    attention_weights = tf.nn.softmax(score, axis=1)        # (B, T, 1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)  # (B, D)

    return context_vector, attention_weights


def build_raw_RNN_model(lstm_units=256,
                        n_hidden=64,
                        n_classes=5,
                        nc=64,
                        T=640,
                        drop_rate=0.5,
                        bidirectional=False,
                        attention=False,
                        attention_units=8):
    """ Input (N, nc=64, T=640), treat (64, 640)) as 2D image and use CNN.
    Ref: https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM.py

    """

    inputs = Input(shape=(nc, T))
    x = tf.transpose(inputs, perm=[0, 2, 1])  # (_, 640, 64)

    if bidirectional:
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout=drop_rate))(x)
        x = Bidirectional(LSTM(lstm_units, return_sequences=attention, recurrent_dropout=drop_rate))(x)
    else:
        x = LSTM(lstm_units, return_sequences=True, recurrent_dropout=drop_rate)(x)
        x = LSTM(lstm_units, return_sequences=attention, recurrent_dropout=drop_rate)(x)

    if attention:
        x, _ = bahdanau_attention(x, attention_units)
        x = Dropout(drop_rate)(x)

    x = Dense(n_hidden)(x)
    x = BatchNormalization(x)
    x = tf.math.softplus(x)
    x = Dropout(drop_rate)(x)

    x = Dense(n_classes)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['acc'])

    return model
