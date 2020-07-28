#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


def BiGRU(Input, max_time, n_input, gru_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''

    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of BiGRU Model
        n_input: The input signal size at one time
        gru_size: The number of GRU units inside the BiGRU Model
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer

    Returns:
        FC_2: Final prediction of BiGRU Model
        FC_1: Extracted features from the first fully connected layer

    '''

    # reshaped Input EEG signals
    Input = tf.reshape(Input, [-1, max_time, n_input])

    # Forward and Backward GRU Model (BiGRU Model)
    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=gru_size)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=gru_size)

    # Dropout for forward and backward GRU Model
    gru_fw_drop = tf.contrib.rnn.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=keep_prob)
    gru_bw_drop = tf.contrib.rnn.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=keep_prob)

    # One layer BiGRU Model
    outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(gru_fw_drop, gru_bw_drop, Input, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    outputs = outputs[:, max_time - 1, :]

    # First fully-connected layer
    FC_1 = tf.matmul(outputs, weights_1) + biases_1
    FC_1 = tf.layers.batch_normalization(FC_1, training=True)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)

    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)

    return FC_2, FC_1
