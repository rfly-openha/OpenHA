# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


import numpy as np
import scipy


def interpolation(
    x: np.ndarray, y: np.ndarray, x_est: np.ndarray, method: str = 'linear'
) -> np.ndarray:
    '''
    Interpolation

    The value of the original data at `x_est` is estimated from the input `x`, `y` and the specified interpolation algorithm.
    The available interpolation algorithms include 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic'; 'previous' and 'next'
        Simply use the preamble or subsequent samples as the current data point;

    Args:
        x: np.ndarray, One-dimensional sequences of real numbers
        y: np.ndarray, Sequence associated with `x`, the length the same as `x`
        x_est: np.ndarray, Sample points to be found
        method: str, Interpolation method

    Returns:
        Sequence of length consistent with `x_est`

    '''
    f = scipy.interpolate.interp1d(x, y, kind=method)
    return f(x_est)


if __name__ == '__main__':
    print('Please use me as a module!')
