# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

from random import shuffle


def split_dataset(X: list, Y: list, percent: float = 0.15) -> tuple:
    '''
    Split dataset randomly.

    Split the dataset with samples `X` and labels `Y` randomly into X1, X2, Y1, Y2.
    The ratio of the numbers of samples in X1 and that of X is determined by `percent`.

    Args:
        X: list, all the samples
        Y: list, labels of each smaples
        percent: float, optional, the ratio to split, 15% for default.

    Returns:
        A tuple, namely (X1, Y1, X2, Y2)

    '''

    # number of the samples in X
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


if __name__ == '__main__':
    print('Please use me as a module!')
