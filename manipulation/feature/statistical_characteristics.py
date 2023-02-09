# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import math
from typing import Union

import numpy as np


def raw_moment(x: np.ndarray, k: int) -> Union[float, np.ndarray]:
    '''
    Compute the raw moment.

    Return the raw moment of all the samples.

    Args:
        x: np.ndarray, whose shape is (n, m), refer to `n` samples and the length of each is `m`.
        k: int, number of the order.

    Returns:
        Raw moment, a scalar if `x.shape==(n,)`, else an ndarray object whose shape is (m,).

    '''

    # initialize a variable
    y = 0
    # sum of the powers of each sample
    for i in x:
        y = y + i**k
    # divide by the number of samples
    y = y / len(x)
    # return the final result

    return y


def central_moment(x: np.ndarray, k: int) -> Union[float, np.ndarray]:
    '''
    Compute the central moment.

    Return the central moment of all the samples.

    Args:
        x: np.ndarray, whose shape is (n, m), refer to `n` samples and the length of each is `m`.
        k: int, number of the order.

    Returns:
        Central moment, a scalar if `x.shape==(n,)`, else an ndarray object whose shape is (m,).

    '''

    # initialize a variable
    y = 0
    # mean of all the samples
    x_bar = x.mean(axis=0)
    # sum of powers of the bias
    for i in x:
        y = y + (i - x_bar) ** k
    # divide by the number of samples
    y = y / len(x)

    return y


def average_rectified_value(x: np.ndarray) -> float:
    '''
    Compute the average rectified value.

    In electrical engineering, the average rectified value of a quantity is the average of its absolute value.

    Args:
        x: np.ndarray, n samples of a quantity.

    Returns:
        The average rectified value.

    '''
    return abs(x).mean(axis=0)


def skewness(x: np.ndarray) -> float:
    '''
    Compute the skewness of time series.

    Args:
        x: np.ndarray, n samples of time series.

    Returns:
        Skewness
    '''
    # 3rd central moment
    mu_3 = central_moment(x, 3)
    # variance
    sigma_square = variance(x)
    # skewness
    skew = mu_3 / math.sqrt(sigma_square) ** 3


def kurtosis(x) -> float:
    '''
    Compute the kurtosis of time series.

    Args:
        x: np.ndarray, n samples of times series.

    Returns:
        Kurtosis
    '''
    # 4th central moment
    mu_4 = central_moment(x, 4)
    # variance
    sigma_square = variance(x)
    # kurtosis
    kurt = mu_4 / sigma_square**2
    return kurt


def variance(x: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Compute the variance.

    Returns the variance of the array elements, a measure of the spread of a distribution.

    Args:
        x: np.ndarray, whose shape is (n, m), refer to `n` samples and the length of each is `m`.

    Returns:
        Variance, a scalar if `x.shape==(n,)`, else an ndarray object whose shape is (m,).

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
