""" Models according to input type
** Input is raw EEG (n_channels, n_steps)
1. raw_cnn: treat input as 2D image, use CNN to process
2. raw_rnn: use RNN to process step-to-step, choice of stacked LSTM/biLSTM, with/without attention
3. raw_transformer: use transformer encoder

** Input is 2D mesh.  Convert (n_channels, 1) to 2D mesh (H=10, W=11) to preserve spatial location
mapping of 64 electrodes
4. mesh_cascade: use CNN to extract spatial features, then pass results to LSTM to extract temporal features
5. mesh_parallel: 2D mesh for CNN, and 1D vector for RNN, they work in parallel.
                  The outputs of both are concatenated before final dense and softmax

** Input is spectral representation
6. spectral_cnn: TODO

** Input is graph representation
7. gnn: TODO
"""

import os
import pickle
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks

tf.compat.v1.disable_eager_execution()

model_name = 'raw_transformer'  # raw_cnn, raw_rnn, raw_transformer, mesh_cascade, mesh_parallel, spectral_cnn, ...

if model_name == 'raw_rnn':
    attention = False
    bidirectional = False
    from models.raw_RNN import build_raw_RNN_model as build_model
elif model_name == 'raw_transformer':
    n_layers = 4
    n_heads = 4
    from models.raw_transformer import build_raw_transformer_model as build_model
elif model_name == 'mesh_cascade':
    from models.mesh_cascade_CNN_RNN import build_mesh_Cascade_CRNN_model as build_model

root = f'./res/{model_name}_' + str(np.datetime64('now'))
if not os.path.exists(root):
    os.makedirs(root)
    os.makedirs(os.path.join(root, 'ckpt'))
    os.makedirs(os.path.join(root, 'history'))

with open('./data/data_mesh_False_segment_False.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']  # raw: (N, nc, T) or mesh: (N, T, 10, 11, 1)
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(X_train.shape)

# Model hyperparameters
n_epochs = 50
drop_rate = 0.75
batch_size = 64

model_dim = 128.0


def make_or_restore_model():
    checkpoints = [root + '/' + name for name in os.listdir(os.path.join(root, 'ckpt'))]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return load_model(latest_checkpoint)
    else:
        print('Creating a new model')

        if model_name == 'raw_rnn':
            n_channels, seq_length = X_train.shape[1:]
            return build_model(input_dim=n_channels, seq_length=seq_length, attention=attention, bidirectional=bidirectional)
        elif model_name == 'raw_transformer':
            n_channels, seq_length = X_train.shape[1:]
            return build_model(input_dim=n_channels, n_layers=n_layers, n_heads=n_heads)
        elif model_name == 'mesh_cascade':
            return build_model()


callbacks_list = [
    callbacks.ModelCheckpoint(os.path.join(root, 'ckpt/ckpt.h5'), save_weights_only=True, save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='acc', patience=10),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    callbacks.TensorBoard(log_dir=os.path.join(root, 'my_log_dir'), histogram_freq=0, write_graph=True, write_images=True)]

if model_name == 'raw_transformer':

    def scheduler(epoch, lr):
        warmup_steps = 4000
        epoch = tf.cast(epoch, tf.float32)
        model_d = tf.cast(model_dim, tf.float32)
        arg1 = tf.math.rsqrt(epoch)
        arg2 = epoch * (warmup_steps ** -1.5)
        return tf.math.rsqrt(model_d) * tf.math.minimum(arg1, arg2)

    callbacks_list.append(callbacks.LearningRateScheduler(schedule=scheduler))

model = make_or_restore_model()

hist = model.fit(X_train,
                 y_train,
                 batch_size=batch_size,
                 epochs=n_epochs,
                 callbacks=callbacks_list,
                 validation_data=(X_val, y_val))

# Save the history
hist_list = glob.glob(os.path.join(root, 'history/*'))
count = len(hist_list)
FILE_NAME = os.path.join(root, f'history/history{str(count)}.pkl')

with open(FILE_NAME, 'wb') as object:
    pickle.dump(hist.history, object)