# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.
from types import FunctionType


def profust_reliability(s: list, msf: FunctionType) -> list[float]:
    '''

    compute the profust reliability

    This function is referred to Corollary 3.2 of Zhao Zhiyao's dissertation P.55, i.e. Equation (3.42).
    Assuming that the system is at S_k at moment t_0 and is in state S_j at moment t, the
    The profust reliablity of the system at moment t is
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]

    Args:
        s: array of states
        msf: Membership Function

    Returns:
        array of profust reliability of each state

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

    计算率模健康度

    This function is referred to Corollary 3.2 of Zhao Zhiyao's dissertation P.55, i.e. Equation (3.42).
    Assuming that the system is at S_k at moment t_0 and is in state S_j at moment t, the
    The profust reliablity of the system at moment t is
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]。
    different from `profust_reliability()`, this function take degree of membership as arguments.

    Args:
        dom: array of degree of membership

    Returns:
        array of profust reliability of each state

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


if __name__ == '__main__':
    print('Please use me as a module!')
