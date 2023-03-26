# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import numpy as np


def local_outlier_factor(points: np.ndarray, k: int) -> list:
    '''
    The local outlier factor algorithm

    Compute the local outlier factors of all the points outliers by the local outlier factor.

    Args:
        points: np.ndarray. An array of vectors of length `n`, namely `n` points.
        k: int. Number of neighbors for k-distance and k-neighbors.

    Returns:
        The local outliers factors of all the points, spcified as an array of positive numeric scalar of length `n`.
    '''

    # the number of points
    n = len(points)
    # distances between all the points
    d = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d[i][j] = np.linalg.norm(points[i] - points[j])
            d[j][i] = d[i][j]

    # k-neighbors
    # the indexes of points in the k-neighbors of each point should be saved
    N_p = []
    # k-distance of each points
    k_distance = [0] * n

    for i in range(n):
        # sort to find the k-distance
        # (distance, corresponding index)
        neighbor_i = [(d[i][j], j) for j in range(n)]
        # sort by distance
        neighbor_i = sorted(neighbor_i, key=lambda x: x[0])
        k_distance[i] = neighbor_i[k][0]
        # find all points within this distance
        for j in range(n - 1, -1, -1):
            if neighbor_i[j][0] <= k_distance[i]:
                break
        N_p.append([x[1] for x in neighbor_i[1 : j + 1]])

    # local reachablity density
    lrd = np.zeros((n,))

    for i in range(n):
        for j in N_p[i]:
            # sum of the rechability distances between i-th point and its neighbors
            lrd[i] += max(k_distance[j], d[i][j])
        # reciprocal of the mean of rechability distances
        lrd[i] = len(N_p[i]) / lrd[i]

    # Local outlier
    lof = np.zeros((n,))

    for i in range(n):
        lof[i] = lrd[N_p[i]].mean() / lrd[i]

    return lof


if __name__ == '__main__':
    print('Please use me as a module!')
