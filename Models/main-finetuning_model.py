#!/usr/bin/env python3
# encoding=utf-8

"""
@Author: Bruce Shuyue Jia
@Date: Nov 5, 2021

This is a code only for reference - two-class classification
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

print("TensorFlow version:", tf.__version__)

# Read Training Data - Target Domain
train_data = pd.read_csv('../target/T-traindata.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels - Target Domain
train_labels = pd.read_csv('../target/T-trainlabel.csv', header=None)
train_labels = np.array(train_labels).astype('float32')
train_labels = tf.one_hot(indices=train_labels, depth=2)
train_labels = np.squeeze(train_labels)

# Read Testing Data - Target Domain
test_data = pd.read_csv('../target/T-testdata.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels - Target Domain
test_labels = pd.read_csv('../target/T-testlabel.csv', header=None)
test_labels = np.array(test_labels).astype('float32')
test_labels = tf.one_hot(indices=test_labels, depth=2)
test_labels = np.squeeze(test_labels)


class CatgoricalTP(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_tp', **kwargs):
        super(CatgoricalTP, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        values = tf.equal(tf.cast(y_pred, 'int32'), tf.cast(y_true, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weights = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weights)

        self.tp.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.tp

    def reset_states(self):
        self.tp.assign(0.)


# Transformer Model
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

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
        self.maxlen = maxlen
        self.embed_dim = embed_dim

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, self.maxlen, self.embed_dim])
        out = x + positions

        return out


maxlen = 3
embed_dim = 97  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer


def get_model():
    # Create a simple model.
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
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# Load the pre-trained model
pretrained_model = get_model()
pretrained_model.load_weights('../transformer/model_weight')

# Make sure the original model will not be updated
# i.e., freezen the model parameters
pretrained_model.trainable = False


# Show the model architecture
pretrained_model.summary()
print('\n\n\n\n')

# Remove the last three layers
num_layer = len(pretrained_model.layers)
pretrained_part = tf.keras.models.Sequential(pretrained_model.layers[0:(num_layer - 3)])
pretrained_part.summary()
print('\n\n\n\n')


# Add some new layers for fine tuning
add_layers = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])


# Build a new model 
rebuild_model = tf.keras.models.Sequential([
    pretrained_part,
    add_layers
])

rebuild_model.build((maxlen * embed_dim,))
rebuild_model.summary()
print('\n\n\n\n')

# Train and evaluate the new model
model = rebuild_model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy", CatgoricalTP()])

history = model.fit(
    train_data, train_labels, batch_size=64, epochs=100, validation_data=(test_data, test_labels)
)
model.evaluate(test_data, test_labels, verbose=2)

