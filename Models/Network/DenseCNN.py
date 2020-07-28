#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf
from Models.Initialize_Variables.Initialize import *


def DenseCNN(Input, keep_prob):
    '''

    Args:
        Input: The reshaped input EEG signals
        keep_prob: The Keep probability of Dropout

    Returns:
        prediction: Final prediction of DenseNet Model

    '''

    # Input reshaped EEG signals: shape 4096 --> 64 X 64
    x_Reshape = tf.reshape(tensor=Input, shape=[-1, 64, 64, 1])

    # First Dense Block
    # First Convolutional Layer
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1_BN = tf.layers.batch_normalization(x_Reshape, training=True)
    h_conv1_Acti = tf.nn.leaky_relu(h_conv1_BN)
    h_conv1 = tf.nn.conv2d(h_conv1_Acti, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1  # 32 feature maps

    # Second Convolutional Layer
    W_conv2 = weight_variable([3, 3, 33, 64])
    b_conv2 = bias_variable([64])
    h_conv2_res = tf.concat([h_conv1, x_Reshape], axis=3)  # 33 feature maps now == 32 + 1
    h_conv2_BN = tf.layers.batch_normalization(h_conv2_res, training=True)
    h_conv2_Acti = tf.nn.leaky_relu(h_conv2_BN)
    h_conv2 = tf.nn.conv2d(h_conv2_Acti, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2  # 64 feature maps

    # First Max Pooling Layer: shape 64 X 64 --> 32 X 32
    h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second Dense Block
    # Third Convolutional Layer
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3_BN = tf.layers.batch_normalization(h_pool1, training=True)
    h_conv3_Acti = tf.nn.leaky_relu(h_conv3_BN)
    h_conv3 = tf.nn.conv2d(h_conv3_Acti, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3  # 128 feature maps

    # Fourth Convolutional Layer
    W_conv4 = weight_variable([3, 3, 192, 256])
    b_conv4 = bias_variable([256])
    h_conv4_res = tf.concat([h_conv3, h_pool1], axis=3)  # 192 feature maps now == 128 + 64
    h_conv4_BN = tf.layers.batch_normalization(h_conv4_res, training=True)
    h_conv4_Acti = tf.nn.leaky_relu(h_conv4_BN)
    h_conv4 = tf.nn.conv2d(h_conv4_Acti, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4  # 256 feature maps

    # First Max Pooling Layer: shape 32 X 32 --> 16 X 16
    h_pool2 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Third Dense Block
    # Fifth Convolutional Layer
    W_conv5 = weight_variable([3, 3, 256, 256])
    b_conv5 = bias_variable([256])
    h_conv5_BN = tf.layers.batch_normalization(h_pool2, training=True)
    h_conv5_Acti = tf.nn.leaky_relu(h_conv5_BN)
    h_conv5 = tf.nn.conv2d(h_conv5_Acti, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5

    # Sixth Convolutional Layer
    W_conv6 = weight_variable([3, 3, 512, 512])
    b_conv6 = bias_variable([512])
    h_conv6_res = tf.concat([h_conv5, h_pool2], axis=3)  # 512 feature maps now == 256 + 256
    h_conv6_BN = tf.layers.batch_normalization(h_conv6_res, training=True)
    h_conv6_Acti = tf.nn.leaky_relu(h_conv6_BN)
    h_conv6 = tf.nn.conv2d(h_conv6_Acti, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6  # 512 feature maps now == 256 + 256

    # Third Max Pooling Layer
    h_pool3 = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten Layer
    h_pool6_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 512])

    # First Fully Connected Layer
    W_fc1 = weight_variable([8 * 8 * 512, 512])
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
