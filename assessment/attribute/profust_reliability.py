# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.
from types import FunctionType

import numpy as np


def profust_reliability(s: list, msf: FunctionType = None) -> list[float]:
    '''

    Compute the profust reliability according to the research results in [1].

    This function refers to Corollary 3 in [1], namely Equation (4).
    Assuming that the system is at S_k at time t_0 and is in state S_j at time t, the
    The profust reliablity of the system at time t is
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]

    Args:
        s: list | np.ndarray. An array of systems states if `msf` is not `None`.
        msf: function, optional. The membership function or `None`.

    References:
        [1] Z. Zhao, Q. Quan, K.-Y. Cai, "A modified profust-performance-reliability algorithm and its application to dynamic systems," Journal of Intelligent & Fuzzy Systems, vol. 32, no. 1, pp. 643-660, 2017. DOI: 10.3233/JIFS-152544.
    '''

    # the variable to save the profust reliability
    R = [0] * len(s)
    # s is an array of system states if msf is not None
    if msf is not None:
        dom = [msf(i) for i in s]
    else:
        dom = s
    # profust reliability of the initial state
    R[0] = dom[0]

    # calculate the profust reliability for each of the remaining states in turn
    for i in range(1, len(dom)):
        # \mu_S(k)
        mu_s_k = dom[i - 1]
        # \mu_S(j)
        mu_s_j = dom[i]
        # Calculating the membership function for fuzzy state transfer
        # \mu_{T_{kj}} = \mu_F(j)-\mu_F(k) = \mu_S(k)-\mu_S(j)
        mu_T = mu_s_k - mu_s_j
        # Definition, P.31 Equation (2.6)
        if mu_T < 0:
            mu_T = 0
        # equation (3.42)
        R[i] = mu_s_k * (1 - mu_T)
    return R


def trapezoidal_membership_func(params: tuple, x: np.ndarray | float):
    '''
    Compute the degree of membership of the state `x` to a set, whose membership function is trapezoidal.

    The shape of the trapezoidal membership function is specified by the `params`, namely `(a, b, c, d)`.
    It will be triangular if `b == c`, and rectangular if `a == b` and `c == d`.

    Args:
        params: tuple. Spcifies the shape of the trapezoidal membership funciton.
        x: np.ndarray | float. An array of system states.

    '''

    def func(x):
        y = 0
        if x <= a:
            y = 0
        elif x <= c:
            y = (x - a) / (c - a)
        elif x <= d:
            y = 1
        elif x <= b:
            y = (x - b) / (d - b)
        else:
            y = 0
        return y

    # all the parameters
    # a <= b <= c <= d
    a, b, c, d = params

    if hasattr(x, '__iter__'):
        y = []
        for i in x:
            y.append(func(i))
    else:
        y = func(x)
    return y


if __name__ == '__main__':
    print('Please use me as a module!')
