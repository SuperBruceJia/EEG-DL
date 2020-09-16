import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, BatchNormalization, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras import Model, optimizers, callbacks
from tensorflow.keras.models import load_model
from glob import glob
import pickle

root = './res/cnn'
if not os.path.exists(root):
    os.makedirs(root)
    os.makedirs(os.path.join(root, 'ckpt'))
    os.makedirs(os.path.join(root, 'history'))
    os.makedirs(os.path.join(root, 'metrics'))

with open('./data/data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']  # (N, nc, T)
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Model hyperparameters
n_epochs = 300
drop_rate = 0.75
batch_size = 64
n_batches = len(X_train) // batch_size


def build_model(n_channels=64, n_samples=64, drop_rate=0.25):

    inputs = Input(shape=(n_channels, n_samples))
    x = tf.expand_dims(inputs, -1)

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


def make_or_restore_model():
    checkpoints = [root + '/' + name for name in os.listdir(os.path.join(root, 'ckpt'))]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return load_model(latest_checkpoint)
    else:
        print('Creating a new model')
        return build_model(n_samples=640)


callbacks_list = [
    callbacks.ModelCheckpoint(os.path.join(root, 'ckpt/ckpt.h5'), save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='acc', patience=10),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    callbacks.TensorBoard(log_dir=os.path.join(root, 'my_log_dir'), histogram_freq=0, write_graph=True, write_images=True)]


# Start training
model = make_or_restore_model()

hist = model.fit(X_train,
                 y_train,
                 batch_size=64,
                 epochs=30,
                 callbacks=callbacks_list,
                 validation_data=(X_val, y_val))

# Save the history
hist_list = glob(os.path.join(root, 'history/*'))
count = len(hist_list)
FILE_NAME = os.path.join(root, f'history/history{str(count)}.pkl')

with open(FILE_NAME, 'wb') as object:
    pickle.dump(hist.history, object)


