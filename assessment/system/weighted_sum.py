# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import numpy as np


def analytical_hierarchy_process(
    A: np.ndarray, method: str = 'eigenvector'
) -> tuple[list, float]:
    '''
    Analytical Hierarchy Process (AHP)

    Compute weights of each element according to AHP and the comparison matrix `A`
    The argument `method` is used to decide the way to finally get weights vector.

    Args:
        A: np.ndarray, comparison matrix which is a square matrix.
        method: str, optional, the way to finally compute weights vector.
            `eigenvector` (default), `geometric_mean`, and `arithmetic_mean`.
            `eigenvector`: the eigenvector corresponding to the max eigenvalue.
            `geometric_mean`: the geometric mean of each row of matrix `A`.
            `arithmetic_mean`: the arithmetic mean of each row of matrix `A`.

    Returns:
        tuple: a list of weights and coherence ratio

    '''
    # size of the matrix `A`
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
    # coherence index
    CI = (lambda_max - n) / (n - 1)
    # coherence ratio
    CR = CI / RI[n - 1]

    # default take the eigenvector as the weights vector
    W = vec[:, [0]]
    if method == 'geometric_mean':
        # mean of each row
        W = A.prod(axis=1) ** (1 / n)
    elif method == 'arithmetic_mean':
        # normlization by column and mean of each row
        W = (A / A.sum(axis=0)).mean(axis=1)
    # normlization
    W /= W.sum()
    # return
    return (W, CI.real, CR.real)


def sum_by_weight(weights: list[float], elements: list[float]) -> float:
    '''
    Compute weighted sum.

    Args:
        weights: list[float], a list of weights of each element.
        elements: list[float], the list of all the elements, whose length is equal to `weights`.

    Returns:
        The weighted sum.
    '''
    # the weighted sum, initialize it with 0
    s = 0
    # add the product of each element and its weight
    for i, j in zip(weights, elements):
        s += i * j

    return s


if __name__ == '__main__':
    print('Please use me as a module!')
