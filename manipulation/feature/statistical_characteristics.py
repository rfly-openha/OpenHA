# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import math

import numpy as np


def raw_moment(x: np.ndarray, k: int) -> float | np.ndarray:
    '''
    Calculates the raw moment.

    Calculates k-th the raw moment of all the observations.

    Args:
        x: np.ndarray. A N-D array of obervations whose shape is `(n, m)`, namely `n` obervations with `m` features.
        k: int. Order of raw moment.

    '''

    # initialize a variable
    y = 0
    # sum of the powers of each observations
    for i in x:
        y = y + i**k
    # divide by the number of observations
    y = y / len(x)
    # return the final result
    return y


def central_moment(x: np.ndarray, k: int) -> float | np.ndarray:
    '''
    Calculates the central moment.

    Calculates the central moment of all the observations.

    Args:
        x: np.ndarray. A N-D array of obervations whose shape is `(n, m)`, namely `n` obervations with `m` features.
        k: int. Order of central moment.
    '''

    # initialize a variable
    y = 0
    # mean of all the obervations
    # along the axis=0
    x_bar = x.mean(axis=0)
    # sum of powers of the bias
    for i in x:
        y = y + (i - x_bar) ** k
    # divide by the number of obervations
    y = y / len(x)

    return y


def average_rectified_value(x: np.ndarray) -> float:
    '''
    Calculates the average rectified value.

    In electrical engineering, the average rectified value of a quantity is the average of its absolute value.

    Args:
        x: np.ndarray. A 1-D array of observations.

    '''
    return abs(x).mean(axis=0)


def skewness(x: np.ndarray) -> float:
    '''
    Compute the skewness of time series.

    Args:
        x: np.ndarray. A N-D array of obervations whose shape is `(n, m)`, namely `n` obervations with `m` features.

    References:
        [1] "Measures of Skewness and Kurtosis," Online, https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm.

    '''
    # 3rd central moment
    mu_3 = central_moment(x, 3)
    # std
    # Note that in computing the kurtosis, the standard deviation is computed using `n` in the denominator rather than `n-1`.
    sigma_square = x.var(axis=0)
    # skewness
    skew = mu_3 / (math.sqrt(sigma_square) ** 3)
    return skew


def kurtosis(x: np.ndarray) -> float:
    '''
    Compute the kurtosis of all the observations.

    Args:
        x: np.ndarray. A N-D array of obervations whose shape is `(n, m)`, namely `n` obervations with `m` features.

    References:
        [1] "Measures of Skewness and Kurtosis," Online, https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm.

    '''
    # 4th central moment
    mu_4 = central_moment(x, 4)
    # std
    # Note that in computing the kurtosis, the standard deviation is computed using `n` in the denominator rather than `n-1`.
    sigma_square = x.var(axis=0)
    # kurtosis
    kurt = mu_4 / (sigma_square**2)
    return kurt


def variance(x: np.ndarray) -> float | np.ndarray:
    '''
    Calculates the variance.

    Calculates the variance of all the observations.

    Args:
        x: np.ndarray. A N-D array of obervations whose shape is `(n, m)`, namely `n` obervations with `m` features.

    '''
    # The `ndarray` object has the method to compute variance.
    # x.var() = \sum{(x-\bar{x})^2}/n
    var = x.var(axis=0)
    # Number of samples
    n = len(x)
    # The unbiased estimate of variance should be \sum{(x-\bar{x})^2}/(n-1)
    return var * n / (n - 1)


if __name__ == '__main__':
    print('Please use me as a module!')
