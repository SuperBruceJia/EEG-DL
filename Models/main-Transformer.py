#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Bruce Shuyue Jia
@Date: Jan 30, 2021
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import os

# 获取当前物理GPU
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.get_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.get_memory_growth = True
# gpu_options = tf.GPUOptions(allow_growth=True)  # 不占满全部显存，按需分配
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)

# Read Training Data
train_data = pd.read_csv('/home/teslav/EEG_DL/training_set.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv('/home/teslav/EEG_DL/training_label.csv', header=None)
train_labels = np.array(train_labels).astype('int')
train_labels = np.squeeze(train_labels)

# Read Testing Data
test_data = pd.read_csv('/home/teslav/EEG_DL/test_set.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv('/home/teslav/EEG_DL/test_label.csv', header=None)
test_labels = np.array(test_labels).astype('int')
test_labels = np.squeeze(test_labels)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, embed_dim])
        out = x + positions
        return out


maxlen = 64  # Only consider 3 input time points
embed_dim = 64  # Features of each time point
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

# Input Time-series
inputs = layers.Input(shape=(maxlen * embed_dim,))
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
x = embedding_layer(inputs)

# Encoder Architecture
transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
x = transformer_block_1(x)
x = transformer_block_2(x)

# Output
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.categorical_accuracy, tf.keras.metrics.Recall()])


callbacks = [keras.callbacks.TensorBoard(update_freq='epoch')]
history = model.fit(
    train_data, train_labels, batch_size=128, epochs=1000, validation_data=(test_data, test_labels), callbacks=callbacks
)

