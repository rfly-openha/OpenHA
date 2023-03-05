# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import numpy as np


def analytical_hierarchy_process(
    A: np.ndarray, method: str = 'eigenvector'
) -> tuple[list, float]:
    '''
    Analytical Hierarchy Process (AHP)

    The Analytic Hierarchy Process is a mathematical model for decision making problems developed by Thomas L. Saaty.
    Compute weight of each element according to pairwise comparison matrix `A`.
    And `method` specifies the way to compute the weights vector.

    Args:
        A: np.ndarray. The n-by-n pairwise comparison matrix.
        method: str, optional. It specifies the way how to compute the weights vector.
            It's specified as `eigenvector` (default), `geometric_mean`, or `arithmetic_mean`.
            `eigenvector`: The eigenvector of `A` corresponding to the maximum eigenvalue.
            `geometric_mean`: The geometric mean of each row of matrix `A`.
            `arithmetic_mean`: The arithmetic mean of each row of matrix `A`.

    Returns:
        A tuple `(W, CI, CR)`, where `W` is the weight vector, `CI` is the consistency index, and `CR` is the consistency ratio.

    '''
    # shape of the matrix `A`
    m, n = A.shape

    # eigenvalues and eigenvectors, respectively.
    d, v = np.linalg.eig(A)

    # the max eigenvalue
    lambda_max = d.max()
    # the eigenvector corresponding to the max eigenvalue
    vec = v[:, [d.argmax()]]
    # default dtype of `vec` is complex, change it to float
    vec.dtype = float

    # Saaty random coherence
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
    # consistency index
    CI = (lambda_max - n) / (n - 1)
    # consistency ratio
    CR = CI / RI[n - 1]

    # default take the eigenvector as the weights vector
    W = vec[:, [0]]
    if method == 'geometric_mean':
        # mean of each row
        W = A.prod(axis=1) ** (1 / n)
    elif method == 'arithmetic_mean':
        # normlize it by column and mean of each row
        W = (A / A.sum(axis=0)).mean(axis=1)
    # normlization
    W /= W.sum()
    # return
    return (W, CI.real, CR.real)


def sum_by_weight(weights: list[float], elements: list[float]) -> float:
    '''
    Calculates the weighted sum.

    Args:
        weights: list[float]. The weight vector.
        elements: list[float]. An array of values of the elements.

    '''
    # the weighted sum, initialize it with 0
    s = 0
    # add the product of each value and its weight
    for i, j in zip(weights, elements):
        s += i * j

    return s


if __name__ == '__main__':
    print('Please use me as a module!')
