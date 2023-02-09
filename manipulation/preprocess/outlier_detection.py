# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.

import numpy as np


def local_outlier_factor(points: np.ndarray, k: int) -> list:
    # LOF 局部离群点因子算法 Local Outlier Factor
    #   MATLAB对LOF算法的实现用于检测离群点
    #
    #   lof = LOF(points, k)
    #
    #   argument
    #       points - 样本点，维度为MxN，表示N个样本点，每个样本点有M个坐标分量
    #       k - 近邻点数量，算法参数之一，应根据样本点数量合适选取
    #   return
    #       lof - 各样本点的局部离群因子
    #

    # 样本点数量
    n = len(points)
    # 两两之间距离计算
    d = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d[i][j] = np.linalg.norm(points[i] - points[j])
            d[j][i] = d[i][j]

    # 保存每个点的k领域内的点的下标
    N_p = []
    # 第k距离
    k_distance = [0] * n

    for i in range(n):
        # 排序找到距离最近的第k个点的距离
        neighbor_i = [(d[i][j], j) for j in range(n)]
        # 按距离进行排序
        neighbor_i = sorted(neighbor_i, key=lambda x: x[0])
        k_distance[i] = neighbor_i[k][0]
        # 以此距离找到所有距离内的点
        for j in range(n - 1, -1, -1):
            if neighbor_i[j][0] <= k_distance[i]:
                break
        N_p.append([x[1] for x in neighbor_i[1 : j + 1]])

    # 局部可达距离
    lrd = np.zeros((n,))

    for i in range(n):
        for j in N_p[i]:
            lrd[i] += max(k_distance[j], d[i][j])
        lrd[i] = len(N_p[i]) / lrd[i]

    # 局部离群因子
    lof = np.zeros((n,))

    for i in range(n):
        lof[i] = lrd[N_p[i]].mean() / lrd[i]

    return lof


if __name__ == '__main__':
    print('Please use me as a module!')
