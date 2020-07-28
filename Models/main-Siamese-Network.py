#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Hide the Configuration and Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import random
import pandas as pd
import numpy as np
import tensorflow as tf
from Models.Network.Siamese_Network import Siamese_Network
from Models.Loss_Function.Loss import loss
from Models.Evaluation_Metrics.Metrics import evaluation
from Models.Initialize_Variables.Initialize import *

# Model Name
Model = 'Siamese_Network_One_Shot_Learning'

# Clear all the stack and use GPU resources as much as possible
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Your Dataset Location, for example EEG-Motor-Movement-Imagery-Dataset
# The CSV file should be named as training_set.csv, training_label.csv, test_set.csv, and test_label.csv
DIR = 'DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/'
SAVE = 'Saved_Files/' + Model + '/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)

def DatasetLoader(DIR):
    '''

    This is the Data Loader for our Library.
    The Dataset was supported via .csv file.
    In the CSV file, each line is a sample.
    For training or testing set, the columns are features of the EEG signals
    For training and testing labels, the columns are corresponding labels.
    In details, please refer to https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow
    to load the EEG Motor Movement Imagery Dataset, which is a benchmark for EEG Motor Imagery.

    Args:
        train_data_1: The training set One
        train_data_2: The training set Two
        train_labels: Training labels
        test_data_1:  The testing set One
        test_data_2:  The testing set One
        test_labels:  Testing labels

    Returns:
        train_data_1: [N_train_1 X M]
        train_data_2: [N_train_2 X M]
        train_labels: [N_train_1 X 1] (N_train_1 == N_train_2)
        test_data_1:  [N_test_1 X M]
        test_data_2:  [N_test_2 X M]
        test_labels:  [N_test_1 X 1] (N_test_1 == N_test_2)
        (N: number of samples, M: number of features)

    '''

    # Read Training Data and Labels
    train_data_1 = pd.read_csv(DIR + 'training_set_1.csv', header=None)
    train_data_1 = np.array(train_data_1).astype('float32')

    train_data_2 = pd.read_csv(DIR + 'training_set_2.csv', header=None)
    train_data_2 = np.array(train_data_2).astype('float32')

    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')

    # Read Testing Data and Labels
    test_data_1 = pd.read_csv(DIR + 'test_set_1.csv', header=None)
    test_data_1 = np.array(test_data_1).astype('float32')

    test_data_2 = pd.read_csv(DIR + 'test_set_2.csv', header=None)
    test_data_2 = np.array(test_data_2).astype('float32')

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')

    return train_data_1, train_data_2, test_data_1, test_data_2, train_labels, test_labels


# Load the dataset, here it uses one-hot representation for labels
train_data_1, train_data_2, test_data_1, test_data_2, train_labels, test_labels = DatasetLoader(DIR=DIR)
train_labels = tf.one_hot(indices=train_labels, depth=4)
train_labels = tf.squeeze(train_labels).eval(session=sess)
test_labels = tf.one_hot(indices=test_labels, depth=4)
test_labels = tf.squeeze(test_labels).eval(session=sess)

# Model Hyper-parameters
num_epoch = 300   # The number of Epochs that the Model run
keep_rate = 0.75  # Keep rate of the Dropout

lr = tf.constant(1e-4, dtype=tf.float32)  # Learning rate
lr_decay_epoch = 50    # Every (50) epochs, the learning rate decays
lr_decay       = 0.50  # Learning rate Decay by (50%)

batch_size = 64
n_batch = train_data_1.shape[0] // batch_size

# Define Placeholders
x_1 = tf.placeholder(tf.float32, [None, 4096])
x_2 = tf.placeholder(tf.float32, [None, 4096])
y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

# Load Model Network
encoding_1 = Siamese_Network(Input=x_1, keep_prob=keep_prob)
encoding_2 = Siamese_Network(Input=x_2, keep_prob=keep_prob)

# Distance
Absolute_distance = tf.abs(encoding_1 - encoding_2)

# Dense Layer to output Probability Score
Weights = weight_variable([15 * 15 * 128, 1])
bias = bias_variable([1])
prediction = tf.nn.sigmoid(tf.matmul(Absolute_distance, Weights) + bias)

# Load Loss Function
loss = loss(y=y, prediction=prediction, l2_norm=True)

# Load Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Load Evaluation Metrics
Global_Average_Accuracy = evaluation(y=y, prediction=prediction)

# Merge all the summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(SAVE + '/train_Writer', sess.graph)
test_writer = tf.summary.FileWriter(SAVE + '/test_Writer')

# Initialize all the variables
sess.run(tf.global_variables_initializer())
for epoch in range(num_epoch + 1):
    # U can use learning rate decay or not
    # Here, we set a minimum learning rate
    # If u don't want this, u definitely can modify the following lines
    learning_rate = sess.run(lr)
    if epoch % lr_decay_epoch == 0 and epoch != 0:
        if learning_rate <= 1e-6:
            lr = lr * 1.0
            sess.run(lr)
        else:
            lr = lr * lr_decay
            sess.run(lr)

    # Randomly shuffle the training dataset and train the Model
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_data_1.shape[0]), batch_size)
        batch_xs_1 = train_data_1[random_batch]
        batch_xs_2 = train_data_2[random_batch]
        batch_ys = train_labels[random_batch]
        sess.run(train_step, feed_dict={x_1: batch_xs_1, x_2: batch_xs_2, y: batch_ys, keep_prob: keep_rate})

    # Show Accuracy and Loss on Training and Test Set
    # Here, for training set, we only show the result of first 100 samples
    # If u want to show the result on the entire training set, please modify it.
    train_accuracy, train_loss = sess.run([Global_Average_Accuracy, loss],
                                          feed_dict={x_1: train_data_1[0:100], x_2: train_data_2[0:100], y: train_labels[0:100], keep_prob: 1.0})
    Test_summary, test_accuracy, test_loss = sess.run([merged, Global_Average_Accuracy, loss],
                                                      feed_dict={x_1: test_data_1, x_2: test_data_2, y: test_labels, keep_prob: 1.0})
    test_writer.add_summary(Test_summary, epoch)

    # Show the Model Capability
    print("Iter " + str(epoch) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(train_accuracy))
    print("Iter " + str(epoch) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
    print("Learning rate is ", learning_rate)
    print('\n')

    # Save the prediction and labels for testing set
    # The "labels_for_test.csv" is the same as the "test_label.csv"
    # We will use the files to draw ROC CCurve and AUC
    # if epoch == num_epoch:
    #     output_prediction = sess.run(prediction, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "prediction_for_test.csv", output_prediction, delimiter=",")
    #     np.savetxt(SAVE + "labels_for_test.csv", test_labels, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()
