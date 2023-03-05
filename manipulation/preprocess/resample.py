# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


import numpy as np
import scipy


def interpolation(
    x: np.ndarray, y: np.ndarray, x_est: np.ndarray, kind: str = 'linear'
) -> np.ndarray:
    '''
    Interpolate a 1-D function.

    `x` and `y` are arrays of values used to approximate some function `f: y = f(x)`.
    This function returns an array of the interpolated values at `x_est`.

    Args:
        x: np.ndarray. A 1-D array of length `n`.

        y: np.ndarray. A N-D array of real values. The length of `y` along the interpolation axis must be equal to the length of `x`.

        x_est: np.ndarray. A 1-D array of points to evaluate the interpolant.

        kind: str, optional. Specifies the kind of interpolation as a string.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order.
        'previous' and 'next' simply return the previous or next value of the point.
        'nearest-up' and 'nearest' differ when interpolating half-integers (e.g., 0.5, 1.5) in that 'nearest-up' rounds up and 'nearest' rounds down.
        Default is 'linear'.


    '''
    f = scipy.interpolate.interp1d(x, y, kind=kind)
    return f(x_est)


if __name__ == '__main__':
    print('Please use me as a module!')
