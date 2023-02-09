# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.
from types import FunctionType


def profust_reliability(s: list, msf: FunctionType) -> list[float]:
    '''

    计算率模健康度

    该函数参考赵峙尧博士论文P.55的推论3.2，即公式(3.42)。
    假设系统在t_0时刻处于S_k，且在t时刻处于状态S_j，则
    系统在t时刻的率模健康度为
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]

    Args:
        s: 状态的列表
        msf: 隶属函数（Membership Function）

    Returns:
        各状态的率模健康度的列表

    '''

    R = [0] * len(s)
    # 初始状态的率模健康度
    R[0] = msf(s[0])
    # 依次计算剩余各状态对应的率模健康度
    for i in range(1, len(s)):
        # \mu_S(k)
        mu_s_k = msf(s[i - 1])
        # \mu_S(j)
        mu_s_j = msf(s[i])
        # 计算模糊状态转移的隶属函数
        # \mu_{T_{kj}} = \mu_F(j)-\mu_F(k) = \mu_S(k)-\mu_S(j)
        mu_T = mu_s_k - mu_s_j

        # 定义，参考P.31公式(2.6)
        if mu_T < 0:
            mu_T = 0

        # 赵峙尧博士论文公式(3.42)
        R[i] = mu_s_k * (1 - mu_T)
    return R


def profust_reliability_dom(dom: list) -> list[float]:
    '''

    计算率模健康度

    该函数参考赵峙尧博士论文P.55的推论3.2，即公式(3.42)。
    假设系统在t_0时刻处于S_k，且在t时刻处于状态S_j，则
    系统在t时刻的率模健康度为
    R(t)=\mu_S(S_k)[1-\mu_T_{SF}(m_{kj})]。
    不同于`profust_reliability()`方法，
    该方法直接接受各状态隶属度作为传入参数。

    Args:
        dom: 状态的隶属度的列表，Degree of Membership

    Returns:
        各状态的率模健康度的列表

    '''
    R = [0] * len(dom)
    # 初始状态的率模健康度
    R[0] = dom[0]
    # 依次计算剩余各状态对应的率模健康度
    for i in range(1, len(dom)):
        # \mu_S(k)
        mu_s_k = dom[i - 1]
        # \mu_S(j)
        mu_s_j = dom[i]
        # 计算模糊状态转移的隶属函数
        # \mu_{T_{kj}} = \mu_F(j)-\mu_F(k) = \mu_S(k)-\mu_S(j)
        mu_T = mu_s_k - mu_s_j
        # 定义，参考P.31公式(2.6)
        if mu_T < 0:
            mu_T = 0
        # 赵峙尧博士论文公式(3.42)
        R[i] = mu_s_k * (1 - mu_T)
    return R


if __name__ == '__main__':
    print('Please use me as a module!')
