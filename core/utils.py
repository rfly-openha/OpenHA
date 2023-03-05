# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

from random import shuffle

import torch
import torch.nn as nn


def split_dataset(X: list, Y: list, percent: float = 0.15) -> tuple:
    '''
    Split the dataset into train and test subsets randomly.

    Split the dataset with features `X` and labels `Y` into `X1, X2, Y1, Y2` randomly.
    The order of the features and their corresponding labels remain unchanged.
    The ratio of the number of features in `X1` to that of `X` is 15% as default.

    It's similar to the function `sklearn.model_selection.train_test_split`.

    Args:
        X: list. An array of feature vectors.
        Y: list. An array of labels.
        percent: float, optional. The proportion of the dataset to include in the train split.

    Returns:
        A tuple, namely (X1, Y1, X2, Y2)

    '''

    # number of the feature vectors in X
    N = len(X)
    # 0~N-1
    index = list(range(N))
    # shuffle the list
    shuffle(index)
    # the exact number, namely the length of X2 and Y2
    N_x1 = round(N * percent)
    # the corresponding elements of the first `N_x1` indexes belong to X1
    # sort the indexes
    sorted_index = sorted(index[0:N_x1])

    # initialize the variables
    X1, X2 = [], []
    Y1, Y2 = [], []

    # the last index that belong to X1
    last = -1
    for i in sorted_index:
        # X1Y1
        X1.append(X[i])
        Y1.append(Y[i])
        # the elements between `last` and `i` belong to X2
        X2.extend(X[last + 1 : i])
        Y2.extend(Y[last + 1 : i])
        # update the last index
        last = i

    # rest of them
    X2.extend(X[last + 1 : N])
    Y2.extend(Y[last + 1 : N])

    return (X1, Y1, X2, Y2)


def load_ANN(net: nn.Module, path: str, **kwargs):
    '''
    Loads an object saved with `torch.save()` or `tf.train.Saver()` from a file.
    Additional args are passed by `kwargs`.

    Args:

        net: nn.Module. An instance of the custom network class, a subclass of `nn.Module`.

        path: str. The path where the file is located.

        **source: 'pytorch' or 'tensorflow'. It specifies the framework in which the files are exported from.

        **map_location: Specifies how to remap storage locations.
        More information is available on https://pytorch.org/docs/stable/generated/torch.load.html.

    '''

    # the platform where these file are from
    source = kwargs.get('source', 'pytorch')
    # a parameter that used to specifies how to remap storage locations
    map_location = kwargs.get('map_location', None)

    if source == 'pytorch':
        # load from local dictionary
        net.load_state_dict(torch.load(path, map_location=map_location))
    else:
        print('waiting for support!')


if __name__ == '__main__':
    print('Please use me as a module!')
