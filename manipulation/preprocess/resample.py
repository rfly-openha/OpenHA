# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


import numpy as np
import scipy


def interpolation(
    x: np.ndarray, y: np.ndarray, x_est: np.ndarray, method: str = 'linear'
) -> np.ndarray:
    '''
    插值

    根据输入`x`、`y`和指定的插值算法，估计原数据在`x_est`处的值。
    可用的插值算法包括'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' 样条插值; 'previous' and 'next'
        简单地以前序或后续样本作为当前数据点;

    Args:
        x: np.ndarray, 一维实数序列
        y: np.ndarray, 与`x`相关的序列，长度与`x`保持一致
        x_est: np.ndarray, 待求的样本点
        method: str, 插值方法

    Returns:
        长度与`x_est`一致的序列

    '''
    f = scipy.linalg.interpolation.interp1d(x, y, kind=method)
    return f(x_est)


if __name__ == '__main__':
    print('Please use me as a module!')
