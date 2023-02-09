# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import math
from itertools import combinations
from typing import Union

import numpy as np
import scipy
import sympy as sp


def control_allocation(
    n: int, d: Union[list, float], ku: Union[list, float], **kwargs
) -> np.ndarray:
    '''
    根据参数生成对应的控制分配矩阵

    Args:
        n: int, 旋翼的数量
        d: Union[list, float], 旋翼到机体系原点的距离；如果为列表，则依次表示每个旋翼到原点的距离
        ku: Union[list, float], 旋翼的力矩系数与了拉力系数的比值，同理可为列表
        **init_angle: Union[list, float], 旋翼臂#1与机体x轴正方向顺时针夹角，默认为0，弧度制，其余旋翼臂均匀分布；
            为列表时，依次表示每个旋翼臂的旋转角度
        **drct: Union[list, int], 每个旋翼臂的旋转方向，1为逆时针，-1为顺时针，默认为顺逆交替且1号桨为顺时针
        **eta: Union[list, float], 标量或列表，各个旋翼的效率系数，默认为1
        **giveup_yaw: bool, 放弃偏航控制，默认为`False`
        **giveup_height: bool, 放弃高度控制，默认为`False`

    Returns:
        控制分配矩阵B_f

    '''
    # 控制分配矩阵维度为4*n
    bf = np.zeros((4, n))

    # 距离为标量时，扩充为列表
    if hasattr(d, '__iter__') == False:
        d = [d] * n
    # 系数为标量时，扩充为列表
    if hasattr(ku, '__iter__') == False:
        ku = [ku] * n

    # 各个旋翼位置与机体x轴正方向顺时针夹角
    init_angle = kwargs.get('init_angle', 0)
    if hasattr(init_angle, '__iter__'):
        phi = init_angle
    else:
        phi = [init_angle + 2 * math.pi * i / n for i in range(n)]

    # 关键字参数中存在`drct`
    # 否则取默认情况，顺逆交替且#1为顺时针，即-1
    drct = kwargs.get('drct', [1 if i % 2 else -1 for i in range(n)])

    # 各旋翼效率系数
    eta = kwargs.get('eta', 1)
    if hasattr(eta, '__iter__'):
        eta = np.diag(eta)
    else:
        eta = np.eye(n) * eta

    # 计算矩阵
    for i in range(n):
        bf[0][i] = 1
        bf[1][i] = -d[i] * math.sin(phi[i])
        bf[2][i] = d[i] * math.cos(phi[i])
        bf[3][i] = drct[i] * ku[i]
    # 乘以效率系数即为最终控制分配矩阵
    B_f = np.matmul(bf, eta)
    # 判断是否删除最后一列
    buttom = 3 if kwargs.get('giveup_yaw', False) else 4
    head = 1 if kwargs.get('giveup_height', False) else 0
    # 使用切片操作返回矩阵中需要的部分
    return B_f[head:buttom]


def acai(
    bf: np.ndarray,
    fmax: Union[np.ndarray, float],
    fmin: Union[np.ndarray, float],
    G: np.ndarray,
) -> float:
    '''

    计算基于剩余控制能力指标（Available Control Authority Index）的可控度

    该函数参考杜光勋博士论文P.35定理3.3，对应公式(3.17)(3.18)
    ACAI其数学本质为是边界为`fmax`和`fmin`的封闭空间经矩阵`bf`映射后得到的新空间的边界，与其内点`G`最近距离的最小值

    Args:
        bf: ndarray, n*m映射矩阵，针对多旋翼则为其控制分配矩阵
        fmax: Union[ndarray, float], 封闭空间的上界，长度为m的列向量或标量
        fmin: Union[ndarray, float], 封闭空间的下届，长度为m的列向量或标量
        G: ndarray, 长度为n的向量，表示某点的坐标

    Returns:
        点`G`到映射后空间各边界最近距离的最小值，
        控制领域，即定义为ACAI

    '''
    # n虚拟控制量个数，m旋翼数量
    [n, m] = bf.shape

    # 参数`fmax`为标量时，扩充为列表
    if not hasattr(fmax, '__iter__'):
        fmax = np.ones((m, 1)) * fmax
    if not hasattr(fmin, '__iter__'):
        fmin = np.ones((m, 1)) * fmin

    # m列中所有可能的自由度为n-1的组合
    S1 = list(combinations(range(m), n - 1))
    # 组合数量
    sm = len(S1)

    # 移动空间使原点为边界点
    G = G - np.matmul(bf, fmin)
    fmax = fmax - fmin

    # 原空间U_f的中心
    fc = fmax / 2
    # 新空间\Omega的中心
    Fc = np.matmul(bf, fc)

    # 记录到各组边界的最小值
    dmin = np.zeros((sm,))

    for j in range(sm):
        # 选择第j种选择
        choose = S1[j]

        # 矩阵B_{1,j}
        B_1j = sp.Matrix(bf[:, choose])
        # 矩阵B_{2,j}
        B_2j = np.delete(bf, choose, axis=1)
        # B_2下标的对应的各组最大值
        fmax_2 = np.delete(fmax / 2, choose, axis=0)

        # 法向量
        xi = B_1j.T.nullspace()[0]
        xi = sp.matrix2numpy(xi / xi.norm())[:, [0]]
        e = np.matmul(xi.T, B_2j)
        # 计算最小值
        dmin[j] = np.matmul(abs(e), fmax_2) - abs(np.matmul(xi.T, Fc - G))

    # 找到绝对值最小的值
    if min(dmin) >= 0:
        doc = min(dmin)
    else:
        doc = -min(abs(dmin))

    if doc < 1e-10 and doc > -1e-10:
        doc = 0

    return doc


def doc_gramian(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    '''
    计算基于Gramian矩阵的可控度

    该函数参考论文[1]中的公式(2.9)，用于计算线性时不变系统的基于Gramian矩阵的可控度。
    根据定义中关于输入量的能量优化目标的不同，该方法包含了三种可控度。

    Args:
        A: np.ndarray, 线性系统状态转移矩阵
        B: np.ndarray, 线性系统的输入矩阵

    Returns:
        三元组，分别为以控制回原点所需最大的最小控制能量和平均控制能量的可控度

    [1] 杜光勋, 全权. 输入受限系统的可控度及其在飞行控制中的应用[J]. 系统科学与数学, 2014, 34(12): 1578-1594.

    '''
    # 状态矩阵的维度
    # 矩阵`A`应为方阵
    ma, na = A.shape
    # 输入矩阵`B`
    mb, nb = B.shape
    # 可控性矩阵
    Q = np.zeros((mb, nb * na))
    # 构造可控性矩阵[B AB A^2B ... A^(n-1)B]
    # 临时变量便于计算`A`的乘方
    T = B
    # 初始化
    Q[:, 0:nb] = B
    for i in range(1, na):
        T = np.matmul(A, T)
        Q[:, i * nb : (i + 1) * nb] = T
    # Q' * Q
    Q = sp.Matrix(np.matmul(Q.T, Q))
    # 最大控制能量的可控度
    rho1 = min(Q.eigenvals())
    # 最小控制能量的可控度
    rho2 = na / Q.inv().trace()
    # 平均控制能量的可控度
    rho3 = math.pow(Q.det(), 1 / na)
    # 以元组形式返回三个值
    return (rho1, rho2, rho3)


def doc_recovery_region(
    A: np.ndarray, B: np.ndarray, U: tuple, T: float, N: int
) -> float:
    '''
    一种基于恢复域的可控度定义

    根据文献[1][2]中的算法，通过将状态空间离散化的方式，计算基于恢复域的可控度的保守估计值。
    恢复域定义为：在规定时间内和容许控制量下，可以被控制至原点的状态的集合。
    在不同的恢复时间，可控度的值有所不同。
    本方法中，一般来说，离散化间隔越小，准确度越高，但计算复杂度越高。

    Args:
        A: np.ndarray, 线性系统状态空间的状态矩阵
        B: np.ndarray, 线性系统状态空间的输入矩阵
        U: tuple, 二元组，分别表示输入的最小最大值，
        T: float, 向前预测的时间长度
        N: int, 步数，T/N为离散化区间长度

    Returns:
        基于恢复域的可控度保守估计值

    [1] 杨斌先, 杜光勋, 全权, 蔡开元. 输入受限下的可控度分析及其在六旋翼飞行器设计中的应用[C]//第三十二届中国控制会议. 中国陕西西安, 2013.
    [2] Klein G, Jr R E L, Longman R W. Computation of a Degree of Controllability Via System Discretization[J]. Journal of Guidance, Control, and Dynamic, 1982, 5(6): 583-589. DOI: 10.2514/3.19793
    '''
    # 状态输入维度
    n, m = B.shape
    # 离散化区间长度
    s_dT = T / N
    # 矩阵计算
    # 文献[1]中公式(8)
    G = sp.Matrix(A * s_dT).exp()
    # 文献[1]中公式(9)
    t = sp.symbols('t')
    h = sp.Matrix(A * t).exp()
    H = sp.integrate(h, (t, 0, s_dT)) * B
    # 文献[1]中公式(13)
    F = np.zeros((n, N * m))

    for i in range(N):
        F[:, (N - i - 1) * m : (N - i) * m] = G**i * H
    # 文献[1]中公式(15)
    K = -G.inv() ** N * F
    # 计算原空间经线性变换后原点到新空间的边界的距离
    doc = acai(sp.matrix2numpy(K), U[1], U[0], np.zeros((n, 1)))

    return doc


def doc_disturbance_rejection_kang(
    A: np.ndarray, B: np.ndarray, D: np.ndarray, Sw: np.ndarray
) -> float:
    '''
    计算表示抗干扰能力的可控度

    该方法用于计算LTI系统的、表示抗干扰能力的可控度。
    文献[1]中提出了一种用于表示系统抗干扰能力的可控度的度量方法。
    具体地，文中给出了适用于一般系统状态空间表达的可控度计算方法，
    通过求解两个一阶微分方程即可得到计算过程所需的两个重要矩阵。
    针对线性时不变系统，可以证明以上该微分方程与lyapunov方程同解。
    具体证明过程及其他详细信息可以参考文献原文。

    Args:
        A: np.ndarray, 线性系统状态空间的状态矩阵
        B: np.ndarray, 线性系统状态空间的输入矩阵
        D: np.ndarray, 线性系统状态方程的扰动向量
        Sw: np.ndarray, 扰动的协方差矩阵

    Returns:
        表示抗干扰能力的可控度

    [1] Kang O, Park Y, Park Y S, et al. New measure representing degree of controllability for disturbance rejection[J/OL]. Journal of Guidance, Control, and Dynamics, 2009, 32(5): 1658-1661. DOI: 10.2514/1.43864.

    '''

    # 对应文献中公式(16)
    # 求解lyapunov方程AX + XA^H + B * B^H=0
    W = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(B, B.T))
    # 对应文献中公式(17)
    # 求解lyapunov方程AX + XA^H + D * Sw * D^H = 0
    Sigma = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(np.matmul(D, Sw), D.T))
    # 对应公式(18)
    # trace(\Sigma/W)
    doc = np.trace(np.matmul(Sigma, np.linalg.inv(W)))
    # 返回
    return doc


if __name__ == '__main__':
    print('Please use me as a module!')
