import tensorflow as tf
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, BatchNormalization, MaxPooling2D, Flatten, Dense


def build_raw_CNN_model(T=640, drop_rate=0.5):
    """ Input (N, nc=64, T=640), treat (64, 640)) as 2D image and use CNN.
    Ref: https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/CNN.py
    """

    nc = 64
    inputs = Input(shape=(nc, T))
    x = tf.expand_dims(inputs, -1)     # (_, 64, 640, 1)

    # 1
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(drop_rate, noise_shape=[tf.shape(x)[0], 1, 1, tf.shape(x)[-1]])(x)

    # 2
    y = Conv2D(32, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    z = tf.concat([x, y], axis=-1)

    # 3
    z = Conv2D(64, (3, 3), padding='same')(z)
    z = LeakyReLU()(z)
    z = Dropout(drop_rate, noise_shape=[tf.shape(z)[0], 1, 1, tf.shape(z)[-1]])(z)
    z = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(z)

    # 4
    z = Conv2D(64, (3, 3), padding='same')(z)  # valid
    z = BatchNormalization()(z)
    z = LeakyReLU()(z)
    z = Dropout(drop_rate, noise_shape=[tf.shape(z)[0], 1, 1, tf.shape(z)[-1]])(z)

    # 5
    w = Conv2D(64, (3, 3), padding='same')(z)
    w = BatchNormalization()(w)
    w = LeakyReLU()(w)

    p = tf.concat([z, w], axis=-1)

    # 6
    p = Conv2D(128, (3, 3), padding='same')(p)
    p = LeakyReLU()(p)
    p = Dropout(drop_rate, noise_shape=[tf.shape(p)[0], 1, 1, tf.shape(p)[-1]])(p)
    p = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(p)

    p = Flatten()(p)

    # 7
    p = Dense(512)(p)
    p = BatchNormalization()(p)
    p = LeakyReLU()(p)
    p = Dropout(drop_rate)(p)

    # 8
    outputs = Dense(5, activation='softmax')(p)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['acc'])

    return model
