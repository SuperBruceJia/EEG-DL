#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import numpy as np
import pandas as pd


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
        train_data: The training set for your Model
        train_labels: The corresponding training labels
        test_data: The testing set for your Model
        test_labels: The corresponding testing labels
        one_hot: One-hot representations for labels, if necessary

    Returns:
        train_data:   [N_train X M]
        train_labels: [N_train X 1]
        test_data:    [N_test X M]
        test_labels:  [N_test X 1]
        (N: number of samples, M: number of features)

    '''

    # Read Training Data and Labels
    train_data = pd.read_csv(DIR + 'training_set.csv', header=None)
    train_data = np.array(train_data).astype('float32')

    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')
    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # train_labels = np.squeeze(train_labels)
    
    # Read Testing Data and Labels
    test_data = pd.read_csv(DIR + 'test_set.csv', header=None)
    test_data = np.array(test_data).astype('float32')

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # test_labels = np.squeeze(test_labels)

    return train_data, train_labels, test_data, test_labels
