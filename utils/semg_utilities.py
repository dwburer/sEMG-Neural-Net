# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 6
    n_channels = 2
    n_steps = 2500

    # hardcode for two types right now
    # first 100 are cyl, second 100 are hook
    # cyl = 1
    # hook = 2
    # lat = 3
    # palm = 4
    # spher = 5
    # tip = 6
    labels = np.concatenate(
        (
            [[class_id for _ in range(100)] for class_id in range(1, n_class + 1)]
        )
    )

    # Read time-series data
    # channel_files = os.listdir(path_signals)
    # channel_files.sort()
    # n_channels = len(channel_files)
    # posix = len(split) + 5

    files = [
        'cyl_ch1.csv',
        'cyl_ch2.csv',
        'hook_ch1.csv',
        'hook_ch2.csv',
        'lat_ch1.csv',
        'lat_ch2.csv',
        'palm_ch1.csv',
        'palm_ch2.csv',
        'spher_ch1.csv',
        'spher_ch2.csv',
        'tip_ch1.csv',
        'tip_ch2.csv',
    ]

    channel_files = [data_path + filename for filename in files]

    # merge files of different grip types into one long file, per channel
    channels = []
    for num_channel in range(n_channels):
        all_of_channel = [pd.read_csv(channel_files[file * (num_channel * n_channels)],  header=None) for file in range(int(len(channel_files) / n_channels))]

        channels.append(
            (pd.concat(all_of_channel), 'channel_%d' % num_channel)
        )

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))

    i_ch = 0
    for channel_data, channel_name in channels:
        X[:,:,i_ch] = channel_data.as_matrix()
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return 
    return X, labels, list_of_channels

def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def one_hot(labels, n_class = 6):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
    



