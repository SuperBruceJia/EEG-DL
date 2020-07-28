#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf
from Models.Initialize_Variables.Initialize import *


def FullyConvCNN(Input, keep_prob):
    '''

    Args:
        Input: The reshaped input EEG signals
        keep_prob: The Keep probability of Dropout

    Returns:
        prediction: Final prediction of  Model

    '''

    # Input reshaped EEG signals
    x_Reshape = tf.reshape(tensor=Input, shape=[-1, 64, 64, 1])

    # First Convolutional Layer
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.conv2d(x_Reshape, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    h_conv1_Acti = tf.nn.leaky_relu(h_conv1)
    h_conv1_drop = tf.nn.dropout(h_conv1_Acti, keep_prob, noise_shape=[tf.shape(h_conv1_Acti)[0], 1, 1, tf.shape(h_conv1_Acti)[3]])

    # Second Convolutional Layer
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.conv2d(h_conv1_drop, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    h_conv2_BN = tf.layers.batch_normalization(h_conv2, training=True)
    h_conv2_Acti = tf.nn.leaky_relu(h_conv2_BN)

    # Third Convolutional Layer
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3_res = tf.concat([h_conv2_Acti, h_conv1_drop], axis=3)
    h_conv3 = tf.nn.conv2d(h_conv3_res, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
    h_conv3_Acti = tf.nn.leaky_relu(h_conv3)
    h_conv3_drop = tf.nn.dropout(h_conv3_Acti, keep_prob, noise_shape=[tf.shape(h_conv3_Acti)[0], 1, 1, tf.shape(h_conv3_Acti)[3]])

    # First Pooling Layer
    W_pool1 = weight_variable([3, 3, 64, 64])
    b_pool1 = bias_variable([64])
    h_pool1 = tf.nn.conv2d(h_conv3_res, W_pool1, strides=[1, 2, 2, 1], padding='VALID') + b_pool1

    # Fourth Convolutional Layer
    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.conv2d(h_pool1, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4
    h_conv4_BN = tf.layers.batch_normalization(h_conv4, training=True)
    h_conv4_Acti = tf.nn.leaky_relu(h_conv4_BN)
    h_conv4_drop = tf.nn.dropout(h_conv4_Acti, keep_prob, noise_shape=[tf.shape(h_conv4_Acti)[0], 1, 1, tf.shape(h_conv4_Acti)[3]])

    # Fifth Convolutional Layer
    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])
    h_conv5 = tf.nn.conv2d(h_conv4_drop, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5
    h_conv5_BN = tf.layers.batch_normalization(h_conv5, training=True)
    h_conv5_Acti = tf.nn.leaky_relu(h_conv5_BN)

    # Sixth Convolutional Layer
    W_conv6 = weight_variable([3, 3, 128, 128])
    b_conv6 = bias_variable([128])
    h_conv6_res = tf.concat([h_conv5_Acti, h_conv4_drop], axis=3)
    h_conv6 = tf.nn.conv2d(h_conv6_res, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6
    h_conv6_Acti = tf.nn.leaky_relu(h_conv6)
    h_conv6_drop = tf.nn.dropout(h_conv6_Acti, keep_prob, noise_shape=[tf.shape(h_conv6_Acti)[0], 1, 1, tf.shape(h_conv6_Acti)[3]])

    # Second Pooling Layer
    W_pool2 = weight_variable([3, 3, 128, 128])
    b_pool2 = bias_variable([128])
    h_pool2 = tf.nn.conv2d(h_conv6_drop, W_pool2, strides=[1, 2, 2, 1], padding='VALID') + b_pool2

    # Flatten Layer
    h_pool6_flat = tf.reshape(h_pool2, [-1, 15 * 15 * 128])

    # First Fully Connected Layer
    W_fc1 = weight_variable([15 * 15 * 128, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.matmul(h_pool6_flat, W_fc1) + b_fc1
    h_fc1_BN = tf.layers.batch_normalization(h_fc1, training=True)
    h_fc1_Acti = tf.nn.leaky_relu(h_fc1_BN)
    h_fc1_drop = tf.nn.dropout(h_fc1_Acti, keep_prob)

    # Second Fully Connected Layer
    W_fc2 = weight_variable([512, 4])
    b_fc2 = bias_variable([4])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return prediction
