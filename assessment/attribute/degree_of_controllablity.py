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
    Generate the control allocation matrix

    Args:
        n: int, number of propellers
        d: Union[list, float], distance from each propeller to the origin the body coordinate.
        ku: Union[list, float], ratio of the moment factor to the force factor of each propeller.
        **init_angle: Union[list, float], the angle from propeller #1 to the positive direction of the body x-axis by clockwise, default is 0, radian system, the rest of the propellers are evenly distributed；
        **drct: Union[list, int], rotation direction of each propeller, 1 is counterclockwise, -1 is clockwise, propeller #1 is clockwise as default.
        **eta: Union[list, float], Efficiency factor for each rotor, default is 1
        **giveup_yaw: bool, give up controling the yaw, default is `False`.
        **giveup_height: bool, give up controling height, default is `False`.

    Returns:
        control allocation matrix `B_f`

    '''
    # dimension of the matrix is 4*n
    bf = np.zeros((4, n))

    # extends is to an array if it's a scalar
    if hasattr(d, '__iter__') == False:
        d = [d] * n
    # extends is to an array if it's a scalar
    if hasattr(ku, '__iter__') == False:
        ku = [ku] * n

    # angles of each propeller
    init_angle = kwargs.get('init_angle', 0)
    if hasattr(init_angle, '__iter__'):
        phi = init_angle
    else:
        phi = [init_angle + 2 * math.pi * i / n for i in range(n)]

    # `drct` is in the kwargs
    drct = kwargs.get('drct', [1 if i % 2 else -1 for i in range(n)])

    # efficiency factor of each propeller
    eta = kwargs.get('eta', 1)
    if hasattr(eta, '__iter__'):
        eta = np.diag(eta)
    else:
        eta = np.eye(n) * eta

    # compute the matrix
    for i in range(n):
        bf[0][i] = 1
        bf[1][i] = -d[i] * math.sin(phi[i])
        bf[2][i] = d[i] * math.cos(phi[i])
        bf[3][i] = drct[i] * ku[i]
    # Multiply by the efficiency factor to get the final control allocation matrix
    B_f = np.matmul(bf, eta)
    # tell whether to delete the first and last rows
    buttom = 3 if kwargs.get('giveup_yaw', False) else 4
    head = 1 if kwargs.get('giveup_height', False) else 0
    # return the required part of the matrix by slice operation
    return B_f[head:buttom]


def acai(
    bf: np.ndarray,
    fmax: Union[np.ndarray, float],
    fmin: Union[np.ndarray, float],
    G: np.ndarray,
) -> float:
    '''

    Compute the Available Control Authority Index.

    This function refers to Theorem 3.3 of Du Guangxun's phd dissertation in P.35, corresponding to Equation (3.17)(3.18)
    The mathematical essence of ACAI is the minimum of the nearest distance between the boundary of a closed space with boundaries `fmax` and `fmin` and the boundary of the new space obtained by mapping the matrix `bf` to its interior point `G`.

    Args:
        bf: ndarray, n*m reflection matrix, control allocation matrix for multicopters
        fmax: Union[ndarray, float], the upper boundary of enclosed space.
        fmin: Union[ndarray, float], the lower boundary of the enclosed space.
        G: ndarray, A vector of length n, representing the coordinates of a point

    Returns:
        Minimum of the nearest distance from point `G` to each boundary of the mapped space
    '''
    # number of control variables is n, and number of propeller is m
    [n, m] = bf.shape

    # extends the `fmax`
    if not hasattr(fmax, '__iter__'):
        fmax = np.ones((m, 1)) * fmax
    if not hasattr(fmin, '__iter__'):
        fmin = np.ones((m, 1)) * fmin

    # combination from m by n-1
    S1 = list(combinations(range(m), n - 1))
    # number of all the combinations
    sm = len(S1)

    # Move the space so that the origin is the boundary point
    G = G - np.matmul(bf, fmin)
    fmax = fmax - fmin

    # Centre of the original space U_f
    fc = fmax / 2
    # New Space \ Omega's Centre
    Fc = np.matmul(bf, fc)

    # save the minimum value to each group boundary
    dmin = np.zeros((sm,))

    for j in range(sm):
        # j-th option
        choose = S1[j]

        # matrix B_{1,j}
        B_1j = sp.Matrix(bf[:, choose])
        # matrix B_{2,j}
        B_2j = np.delete(bf, choose, axis=1)
        fmax_2 = np.delete(fmax / 2, choose, axis=0)

        # Normal vector
        xi = B_1j.T.nullspace()[0]
        xi = sp.matrix2numpy(xi / xi.norm())[:, [0]]
        e = np.matmul(xi.T, B_2j)
        # Calculate the minimum value
        dmin[j] = np.matmul(abs(e), fmax_2) - abs(np.matmul(xi.T, Fc - G))

    # Find the value with the smallest absolute value
    if min(dmin) >= 0:
        doc = min(dmin)
    else:
        doc = -min(abs(dmin))

    if doc < 1e-10 and doc > -1e-10:
        doc = 0

    return doc


def doc_gramian(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    '''
    Calculating Gramian matrix-based controllability

    The function refers to equation (2.9) in the paper [1] for calculating the Gramian matrix-based controllability of a linear time-invariant system.
    The method incorporates three controllability degrees, depending on the energy optimisation objective of the definition with respect to the input quantities.

    Args:
        A: np.ndarray, state transfer matrix
        B: np.ndarray, input matrix

    Returns:
        triple

    [1] 杜光勋, 全权. 输入受限系统的可控度及其在飞行控制中的应用[J]. 系统科学与数学, 2014, 34(12): 1578-1594.

    '''
    # Dimensionality of the state matrix
    # The matrix `A` should be square
    ma, na = A.shape
    # input matrix`B`
    mb, nb = B.shape
    # Controllability Matrix
    Q = np.zeros((mb, nb * na))
    # Constructing the Controllability Matrix [B AB A^2B ... A^(n-1)B]
    # Temporary variables to save the power of `A`
    T = B
    # initialization
    Q[:, 0:nb] = B
    for i in range(1, na):
        T = np.matmul(A, T)
        Q[:, i * nb : (i + 1) * nb] = T
    # Q' * Q
    Q = sp.Matrix(np.matmul(Q.T, Q))
    # doc based on the maximum control energy
    rho1 = min(Q.eigenvals())
    # doc based on the minimum control energy
    rho2 = na / Q.inv().trace()
    # doc based on the average control energy
    rho3 = math.pow(Q.det(), 1 / na)
    # return them as a triple
    return (rho1, rho2, rho3)


def doc_recovery_region(
    A: np.ndarray, B: np.ndarray, U: tuple, T: float, N: int
) -> float:
    '''
    A definition of controllability based on the recovery domain

    A conservative estimate of the controllability based on the recovery domain is calculated by discretizing the state space according to the algorithm in the literature [1][2].
    The recovery domain is defined as the set of states that can be controlled to the origin in a specified time and with a tolerable amount of control.
    At different recovery times, the value of the controllability varies.
    In this method, in general, the smaller the discretization interval, the higher the accuracy, but the higher the computational complexity.

    Args:
        A: np.ndarray, state transfer matrix of a LTI system
        B: np.ndarray, input matrix of a LTI system
        U: tuple, maximum and minimum value of each input
        T: float, time to predict
        N: int, steps，T/N is the length of discretization

    Returns:
        Conservative estimates of controllability based on recovery domains

    [1] 杨斌先, 杜光勋, 全权, 蔡开元. 输入受限下的可控度分析及其在六旋翼飞行器设计中的应用[C]//第三十二届中国控制会议. 中国陕西西安, 2013.
    [2] Klein G, Jr R E L, Longman R W. Computation of a Degree of Controllability Via System Discretization[J]. Journal of Guidance, Control, and Dynamic, 1982, 5(6): 583-589. DOI: 10.2514/3.19793
    '''
    # dimension of input matrix
    n, m = B.shape
    # length of the discretization interval
    s_dT = T / N
    # Matrix calculation
    # Equation (8) in [1]
    G = sp.Matrix(A * s_dT).exp()
    # Equation (9) in [1]
    t = sp.symbols('t')
    h = sp.Matrix(A * t).exp()
    H = sp.integrate(h, (t, 0, s_dT)) * B
    # Equation (13) in [1]
    F = np.zeros((n, N * m))

    for i in range(N):
        F[:, (N - i - 1) * m : (N - i) * m] = G**i * H
    # Equation (15) in [1]
    K = -G.inv() ** N * F
    # Calculate the distance from the origin of the original space to the boundary of the new space after the linear transformation
    doc = acai(sp.matrix2numpy(K), U[1], U[0], np.zeros((n, 1)))

    return doc


def doc_disturbance_rejection_kang(
    A: np.ndarray, B: np.ndarray, D: np.ndarray, Sw: np.ndarray
) -> float:
    '''
    Calculation of the degree of controllability indicating immunity to interference

    This method is used to calculate the controllability of an LTI system, indicating the resistance to interference.
    A metric for expressing the controllability of a system against interference is presented in [1].
    Specifically, a method is given for calculating the controllability for a general system state space representation.
    The two important matrices required for the calculation process are obtained by solving two first-order differential equations.
    For linear time-invariant systems, it is shown that the differential equation is identical to the lyapunov equation.
    Details of the proof and other details can be found in the original literature.

    Args:
        A: np.ndarray, State matrix of a linear system state space
        B: np.ndarray, Input matrix for linear system state space
        D: np.ndarray, Perturbation vectors for the equation of state of a linear system
        Sw: np.ndarray, Covariance matrix of perturbations

    Returns:
        Indicates controllability of immunity to interference

    [1] Kang O, Park Y, Park Y S, et al. New measure representing degree of controllability for disturbance rejection[J/OL]. Journal of Guidance, Control, and Dynamics, 2009, 32(5): 1658-1661. DOI: 10.2514/1.43864.

    '''

    # Corresponding to equation (16) in the literature
    # solve lyapunovfunciton AX + XA^H + B * B^H=0
    W = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(B, B.T))
    # Corresponding to equation (17) in the literature
    # solve lyapunov function AX + XA^H + D * Sw * D^H = 0
    Sigma = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(np.matmul(D, Sw), D.T))
    # equation (18)
    # trace(\Sigma/W)
    doc = np.trace(np.matmul(Sigma, np.linalg.inv(W)))
    return doc


if __name__ == '__main__':
    print('Please use me as a module!')
