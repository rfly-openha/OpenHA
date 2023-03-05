# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import math
from itertools import combinations

import numpy as np
import scipy
import sympy as sp


def control_allocation(
    n: int, d: list | float, ku: list | float, **kwargs
) -> np.ndarray:
    '''
    Returns the control allocation matrix of a multicopter.

    Args:

        n: int. The number of propellers of a multicopter.

        d: list | float. The distance between each propeller and the origin of the body coordinate.

        ku: list | float. The ratio of the torque to thrust of each propeller.

        **init_angle: list | float. The angle from $O_\text{b}x_\text{b}$ axis to each supporting arm of the propeller in clockwise direction in radians.
        When all the propellers are distributed evenly, specify `init_angle` as a positive value.
        It then indicates the angle of propeller #1, namely $\phi_1$ = `init_angle`.


        **drct: list. The rotation direction of each propeller, specified as an array of length `n`, in which each element is `-1` or `1`. Default is `[1, -1, 1, -1, 1, -1]`.

        **eta: list | float. The efficient coefficient of each propeller. Default is `1`.

        **giveup_yaw: bool. Whether to give up the control of height or not. `True` or `False` (default).

        **giveup_height: bool. Whether to give up the control of yaw or not. `True` or `False` (default).


    '''
    # default shape of the matrix is (4, n)
    bf = np.zeros((4, n))

    # extend it to an array if it's a scalar
    if hasattr(d, '__iter__') == False:
        d = [d] * n
    # extend it to an array if it's a scalar
    if hasattr(ku, '__iter__') == False:
        ku = [ku] * n

    # angles of each propeller
    init_angle = kwargs.get('init_angle', 0)
    if hasattr(init_angle, '__iter__'):
        phi = init_angle
    else:
        phi = [init_angle + 2 * math.pi * i / n for i in range(n)]

    # if `drct` is in the kwargs
    # default is `[1, -1, 1, -1, ...]`
    drct = kwargs.get('drct', [-1 if i % 2 else 1 for i in range(n)])

    # efficiency coefficient of each propeller
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

    # Multiply by the efficiency coefficient to get the control allocation matrix
    B_f = np.matmul(bf, eta)

    # whether to delete the first and last rows
    buttom = 3 if kwargs.get('giveup_yaw', False) else 4
    head = 1 if kwargs.get('giveup_height', False) else 0

    # return the required part of the matrix by slicing
    return B_f[head:buttom]


def acai(
    bf: np.ndarray,
    fmax: np.ndarray | float,
    fmin: np.ndarray | float,
    G: np.ndarray,
) -> float:
    '''

    Compute the DOC based on the Available Control Authority Index (ACAI).

    This function refers to Theorem 3 of paper [1].
    More introduction about this function is avaiable.

    Args:

        bf: ndarray, an n-by-m matrix. It refers to the linear map between two spaces. Specify it as the control allocation matrix in the computation of ACAI.

        fmax: np.ndarray | float. The upper bound of each dimension of the space $U$.

        fmin: np.ndarray | float. The lower bound of each dimension of the space $U$.

        G: a 1-D array of length `n`. An point in $\Omega$.

    References:
        [1] G.-X. Du, Q. Quan, B. Yang, and K.-Y. Cai, "Controllability Analysis for Multirotor Helicopter Rotor Degradation and Failure," Journal of Guidance, Control, and Dynamics, vol. 38, no. 5, pp. 978-984, 2015. DOI: 10.2514/1.G000731.
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

    # centre of the original space U
    fc = fmax / 2
    # new space \Omega's Centre
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

        # normal vector
        xi = B_1j.T.nullspace()[0]
        xi = sp.matrix2numpy(xi / xi.norm())[:, [0]]
        e = np.matmul(xi.T, B_2j)
        # calculate the minimum value
        dmin[j] = np.matmul(abs(e), fmax_2) - abs(np.matmul(xi.T, Fc - G))

    # find the value with the smallest absolute value
    if min(dmin) >= 0:
        doc = min(dmin)
    else:
        doc = -min(abs(dmin))

    if doc < 1e-10 and doc > -1e-10:
        doc = 0

    return doc


def doc_gramian(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    '''
    Computes the Gramian-matrix-based degree of controllability (DOC).

    The function refers to equations (2.9) in [1] and Section 2.2 in [2] to compute the Gramian-matrix-based DOC of a linear time-invariant (LTI) system.
    Three candidates for physically meaningful measures are included.

    Args:
        A: np.ndarray. System transition matrix of the state-space model of an LTI system, specified as an n-by-n square matrix.
        B: np.ndarray, Input coefficient matrix of the state-space model of an LIT system, specified as an n-by-p matrix.

    Returns:
        A tuple `(rho1, rho2, rho3)`, where rho1 is the maximum eigenvalue of $W^{-1}$, rho2 is the trace of it, and rho3 is the determinant of it.
        More information about the matrix $W$ if available in its corresponding document.

    [1] G.-X. Du, Q. Quan, "Degree of Controllability and its Application in Aircraft Flight Control," Journal of Systems Science and Mathematical Sciences, vol. 34, no. 12, pp. 1578-1594, 2014.
    [2] P.C. Müller, H.I. Weber, "Analysis and optimization of certain qualities of controllability and observability for linear dynamical systems," vol. 8, no. 3, pp. 237-246, 1972. DOI: 10.1016/0005-1098(72)90044-1.

    '''

    # Shape of the state matrix
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
    Computes the degree of controllability based on the recovery region.

    A conservative estimate of the DOC based on the recovery region is computed by discretizing the state-space model according to the algorithm in [1][2].
    The recovery region is defined as the set of states that can be controlled to the origin in a specified time and control input.
    The value of the DOC varies based on the recovery time.
    In this method, in general, the smaller the discretization interval is, the higher the accuracy is, but the higher the computational complexity.

    Args:
        A: np.ndarray. System transition matrix of the state-space model of an LTI system.
        B: np.ndarray. System input matrix of the state-space model of an LTI system.
        U: tuple. The maximum and minimum value of control inputs.
        T: float. The recovery time.
        N: int. The prediction step, and T/N is the length of each interval

    References:

        [1] B Yang, G.-X. Du, Q. Quan, K.-Y. Cai, "The Degree of Controllability with Limited Input and an Application for Hexacopter Design," in Proceedings of the 32nd Chinese Control Conference. Xi'an, Shaanxi, China, 2013.
        [2] G. Klein, R. E. L. Jr., W. W. Longman, "Computation of a Degree of Controllability Via System Discretization," Journal of Guidance, Control, and Dynamic, vol 5, no. 6, pp. 583-589, 1982. DOI: 10.2514/3.19793
    '''
    # shape of the input matrix
    n, m = B.shape
    # length of the discretization interval
    s_dT = T / N
    # Matrix calculation
    # equation (8) in [1]
    G = sp.Matrix(A * s_dT).exp()
    # equation (9) in [1]
    t = sp.symbols('t')
    h = sp.Matrix(A * t).exp()
    H = sp.integrate(h, (t, 0, s_dT)) * B
    # equation (13) in [1]
    F = np.zeros((n, N * m))

    # save the power of G
    power_G = H
    # when i = 0
    F[:, N * m - m : N * m] = H
    # otherwise
    for i in range(1, N):
        # left multiply
        power_G = np.matmul(G, power_G)
        F[:, (N - i - 1) * m : (N - i) * m] = power_G
    # omit small items
    F[abs(F) < 1e-6] = 0
    # equation (15) in [1]
    K = -G.inv() ** N * F
    # calculate the distance from the origin of the original space to the boundary of the new space after the linear map
    doc = acai(sp.matrix2numpy(K), U[1], U[0], np.zeros((n, 1)))

    return doc


def doc_disturbance_rejection_kang(
    A: np.ndarray, B: np.ndarray, D: np.ndarray, Sw: np.ndarray
) -> float:
    '''
    Computes the new measure representing degree of controllability for disturbance rejection.

    This function is used to compute the controllability for disturbance rejection of an LTI system.
    A new measure to represent the capabilities of disturbance rejection is proposed in [1].
    More information about the matrix $W$ if available in its corresponding document.

    Args:
        A: np.ndarray. System transition matrix of the state-space model of an LTI system, specified as an n-by-n square matrix.
        B: np.ndarray. Input coefficient matrix of the state-space model of an LIT system, specified as an n-by-r matrix.
        D: np.ndarray. Disturbance matrix, specified as an n-by-l matrix.
        Sw: np.ndarray. Covariance matrix of disturbance vectors, specified as an l-by-l square matrix.

    Returns:
        Degree of controllability for disturbance rejection

    [1] O. Kang, Y. Park, Y. S. Park, M. Suh, "New measure representing degree of controllability for disturbance rejection," Journal of Guidance, Control, and Dynamics, vol. 32, no. 5, pp. 1658-1661, 2009. DOI: 10.2514/1.43864.

    '''

    # Corresponding to equation (16) in the literature
    # solve lyapunovfunciton AX + XA^H + B * B^H=0
    W = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(B, B.T))
    # Corresponding to equation (17) in the literature
    # solve lyapunov function AX + XA^H + D * Sw * D^H = 0
    Sigma = scipy.linalg.solve_continuous_lyapunov(A, -np.matmul(np.matmul(D, Sw), D.T))
    # equation (18)
    # trace(\Sigma/W)
    doc = np.trace(np.matmul(np.linalg.inv(W), Sigma))
    return doc


if __name__ == '__main__':
    print('Please use me as a module!')
