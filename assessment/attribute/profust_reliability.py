# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.
from types import FunctionType


def profust_reliability(s: list, msf: FunctionType) -> list[float]:
    '''

    Compute the profust reliability

    This function refers to Corollary 3 in [1], namely Equation (4).
    Assuming that the system is at S_k at time t_0 and is in state S_j at time t, the
    The profust reliablity of the system at time t is
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]

    Args:
        s: array of states
        msf: the membership function

    Returns:
        array of profust reliability of each state

    References:
        [1] Z. Zhao, Q. Quan, K.-Y. Cai, "A modified profust-performance-reliability algorithm and its application to dynamic systems," Journal of Intelligent & Fuzzy Systems, vol. 32, no. 1, pp. 643-660, 2017. DOI: 10.3233/JIFS-152544.
    '''

    R = [0] * len(s)
    # Initial state of profust health
    R[0] = msf(s[0])
    # Calculate the profust health for each of the remaining states in turn
    for i in range(1, len(s)):
        # \mu_S(k)
        mu_s_k = msf(s[i - 1])
        # \mu_S(j)
        mu_s_j = msf(s[i])
        # Calculating the membership function for fuzzy state transfer
        # \mu_{T_{kj}} = \mu_F(j)-\mu_F(k) = \mu_S(k)-\mu_S(j)
        mu_T = mu_s_k - mu_s_j

        # Definition, P.31 Equation (2.6)
        if mu_T < 0:
            mu_T = 0

        # equation (3.42)
        R[i] = mu_s_k * (1 - mu_T)
    return R


def profust_reliability_dom(dom: list) -> list[float]:
    '''

    Compute the profust reliability

    This function refers to Corollary 3 in [1], namely Equation (4).
    Assuming that the system is at S_k at time t_0 and is in state S_j at time t, the
    The profust reliablity of the system at time t is
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]。
    Different from `profust_reliability()`, this function take degree of membership as an argument rather than system states and membership function.

    Args:
        dom: array of degree of membership of system states.

    Returns:
        array of profust reliability of each state.

    References:
        [1] Z. Zhao, Q. Quan, K.-Y. Cai, "A modified profust-performance-reliability algorithm and its application to dynamic systems," Journal of Intelligent & Fuzzy Systems, vol. 32, no. 1, pp. 643-660, 2017. DOI: 10.3233/JIFS-152544.

    '''
    R = [0] * len(dom)
    # Initial state of profust health
    R[0] = dom[0]
    # Calculate the profust health for each of the remaining states in turn
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


def trapezoidal_membership_func(params: tuple, x: float):
    '''
    Compute the degree of membership of the state `x` by the trapezoidal membership function.

    The shape of the trapezoidal membership function is determined by the parameter `params`, namely `(a, b, c, d)`.
    It will be a triangular membership function when `b == c`.

    Args:
        params: tuple, determine the shape of the trapezoidal membership funciton.
        x: float, system state.

    Returns:
        The degree of membership of the system state `x`.
    '''
    # degree of membership
    y = 0
    # all the parameters
    # a <= b <= c <= d
    a, b, c, d = params

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


if __name__ == '__main__':
    print('Please use me as a module!')
