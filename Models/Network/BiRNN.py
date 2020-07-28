#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


def BiRNN(Input, max_time, n_input, rnn_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''

    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of BiRNN Model
        n_input: The input signal size at one time
        rnn_size: The number of RNN units inside the BiRNN Model
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer

    Returns:
        FC_2: Final prediction of BiRNN Model
        FC_1: Extracted features from the first fully connected layer

    '''

    # reshaped Input EEG signals
    Input = tf.reshape(Input, [-1, max_time, n_input])

    # Forward and Backward RNN Model (BiRNN Model)
    rnn_fw_cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
    rnn_bw_cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)

    # Dropout for forward and backward RNN Model
    rnn_fw_drop = tf.contrib.rnn.DropoutWrapper(cell=rnn_fw_cell, input_keep_prob=keep_prob)
    rnn_bw_drop = tf.contrib.rnn.DropoutWrapper(cell=rnn_bw_cell, input_keep_prob=keep_prob)

    # One layer BiRNN Model
    outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(rnn_fw_drop, rnn_bw_drop, Input, dtype=tf.float32)
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
