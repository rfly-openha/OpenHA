# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import numpy as np


def local_outlier_factor(points: np.ndarray, k: int) -> list:
    '''
    The local outlier factor algorithm

    Detect outliers by the local outlier factor.

    Args:
        points: np.ndarray, samples points of m-by-n, representing `m` samples.
        k: int, number of nearest neighbours, one of the parameters of the algorithm, which should be selected appropriately according to the number of sample points.

    Returns:
        lof, a list of local outlier factors of each sample points.
    '''

    # Number of sample points
    n = len(points)
    # Calculation of the distance
    d = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d[i][j] = np.linalg.norm(points[i] - points[j])
            d[j][i] = d[i][j]

    # Save the subscripts of the points in the k-field of each point
    N_p = []
    # k-th distance
    k_distance = [0] * n

    for i in range(n):
        # Sort to find the distance to the kth nearest point
        neighbor_i = [(d[i][j], j) for j in range(n)]
        # Sort by distance
        neighbor_i = sorted(neighbor_i, key=lambda x: x[0])
        k_distance[i] = neighbor_i[k][0]
        # Find all points within this distance
        for j in range(n - 1, -1, -1):
            if neighbor_i[j][0] <= k_distance[i]:
                break
        N_p.append([x[1] for x in neighbor_i[1 : j + 1]])

    # Local reachable distance
    lrd = np.zeros((n,))

    for i in range(n):
        for j in N_p[i]:
            lrd[i] += max(k_distance[j], d[i][j])
        lrd[i] = len(N_p[i]) / lrd[i]

    # Local outlier
    lof = np.zeros((n,))

    for i in range(n):
        lof[i] = lrd[N_p[i]].mean() / lrd[i]

    return lof


if __name__ == '__main__':
    print('Please use me as a module!')
