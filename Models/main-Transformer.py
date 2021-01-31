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

# Read Training Data
train_data = pd.read_csv('training_set.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv('training_label.csv', header=None)
train_labels = np.array(train_labels).astype('float32')
train_labels = np.squeeze(train_labels)

# Read Testing Data
test_data = pd.read_csv('test_set.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv('test_label.csv', header=None)
test_labels = np.array(test_labels).astype('float32')
test_labels = np.squeeze(test_labels)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.50):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation=tf.nn.leaky_relu), layers.Dense(embed_dim), ])
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
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.expand_dims(x, axis=2)
        out = x + positions
        return out


maxlen = 97  # Only consider the first 97 time points
embed_dim = 1  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 16  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)

# Input Time-series
x = embedding_layer(inputs)

# Encoder Architecture with 6 Blocks
transformer_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block_1(x)

# transformer_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block_2 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block_3 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block_4 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block_5 = TransformerBlock(embed_dim, num_heads, ff_dim)
# transformer_block_6 = TransformerBlock(embed_dim, num_heads, ff_dim)
# x = transformer_block_1(x)
# x = transformer_block_2(x)
# x = transformer_block_3(x)
# x = transformer_block_4(x)
# x = transformer_block_5(x)
# x = transformer_block_6(x)

# Output
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation=tf.nn.leaky_relu)(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_data, train_labels, batch_size=128, epochs=1000, validation_data=(test_data, test_labels)
)
