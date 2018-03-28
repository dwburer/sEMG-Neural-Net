# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 2
    n_steps = 2500

    # Paths
    # path_ = os.path.join(data_path, split)
    # path_signals = os.path.join(path_, "Inertial_Signals")

    # # Read labels and one-hot encode
    # label_path = os.path.join(path_, "y_" + split + ".txt")
    # labels = pd.read_csv(label_path, header = None)

    # hardcode for two types right now
    # first 100 are cyl, second 100 are hook
    # cyl = 1
    # hook = 2
    labels = np.concatenate(
        (
            [1 for i in range(100)],
            [2 for i in range(100)]
        )
    )

    # Read time-series data
    # channel_files = os.listdir(path_signals)
    # channel_files.sort()
    # n_channels = len(channel_files)
    # posix = len(split) + 5


    channel_files = [
        data_path + 'cyl_ch1.csv',
        data_path + 'cyl_ch2.csv',
        data_path + 'hook_ch1.csv',
        data_path + 'hook_ch2.csv'
    ]
    n_channels = 2

    # merge files of different grip types into one long file, per channel
    all_channel_1 = (
        pd.concat([pd.read_csv(channel_files[0],  header=None), pd.read_csv(channel_files[2],  header=None)]),
        'channel_1'
    )

    all_channel_2 = (
        pd.concat([pd.read_csv(channel_files[1],  header=None), pd.read_csv(channel_files[3],  header=None)]),
        'channel_2'
    )

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))



    # i_ch = 0
    # for fil_ch in channel_files:
    #     channel_name = fil_ch[:-posix]
    #     dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
    #     X[:,:,i_ch] = dat_.as_matrix()

    #     # Record names
    #     print(channel_name)
    #     list_of_channels.append(channel_name)

    #     # iterate
    #     i_ch += 1


    i_ch = 0
    for channel_data, channel_name in [all_channel_1, all_channel_2]:
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

def one_hot(labels, n_class = 2):
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
    



