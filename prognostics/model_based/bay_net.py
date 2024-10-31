# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.
# Author: Juanyi Zhou


import pandas as pd
import numpy as np
import random
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination, BeliefPropagation


class Triple:
    """Triple data structure"""
    def __init__(self):
        self.row = None     # The row where the non-zero element is located
        self.col = None     # The column where the non-zero elements are located
        self.data = None       # Element value, for example: 1


class Matrixinfo:
    """Correlation Matrix Information"""
    def __init__(self, Matrix_path):
        self.rows = 0  # Number of matrix rows
        self.cols = 0  # Matrix columns
        self.nums = 0  # The number of non-zero values ​​in the matrix
        self.dm = None
        self.tripleList = []  # Triple(),FxT
        self.matrix_to_triple(Matrix_path)  # Perform the conversion of the original D matrix into a triplet

    def matrix_to_triple(self, Matrix_path):
        """
        D matrix is ​​converted into triple form
        :param MatrixInfo: D Matrix Information Class
        :param Matrix_path: D matrix file path
        :return:
        """
        # dataFrame
        matrixData = pd.read_excel(Matrix_path, sheet_name='Sheet1', index_col=0)
        # Convert to Matrix
        dm = matrixData.values
        self.dm = dm

        # View Matrix Dimensions
        self.rows = matrixData.shape[0]
        self.cols = matrixData.shape[1]

        # Traverse and convert to triplets
        for i in range(len(dm)):
            for j in range(len(dm[0])):
                if dm[i][j] == 1:
                    tempTriple = Triple()
                    tempTriple.row = i
                    tempTriple.col = j
                    tempTriple.data = 1
                    # Add to the triple list
                    self.tripleList.append(tempTriple)
                    self.nums = self.nums + 1


def bay_set_model(path=None, model_path=None):
    """
    Use this function to train the model, and get the correlated files.

    Args:
        path: the file path of the original D matrix.

        model_path: the file path where the `xmlbif` file is located.

    """
    # step1: Read the correlation matrix and prepare modeling information
    m = Matrixinfo(path)
    # step2: Read the original matrix and determine the meaning of the rows and columns representing the nodes
    # If it is later changed to a correlation matrix, the rows and columns are consistent, and the matrix can be read directly
    # It can be merged with step 1 later
    data = pd.read_excel(path, sheet_name='Sheet1', index_col=0)

    faults = data.index.values
    tests = data.columns.values
    # step3: Generate simulation data train
    # Solving Chinese character problems
    nodes_test = []
    nodes_fault = []
    testNum = 0
    faultNum = 0
    for node in tests:
        nodes_test.append('T' + str(testNum))
        testNum = testNum + 1
    for node in faults:
        nodes_fault.append('F' + str(faultNum))
        faultNum = faultNum + 1

    # nodes = tests
    # for x in faults:
    #     nodes = np.append(nodes, x)
    data_size = int(1e5)
    test_data = pd.DataFrame(columns=nodes_test, index=np.arange(data_size))
    fault_data = pd.DataFrame(columns=nodes_fault, index=np.arange(data_size))

    for i in nodes_fault:
        # Configure the probability here
        probability = random.uniform(0.1, 0.4)
        nums = np.random.choice([0, 1], size=data_size, p=[probability, 1 - probability])
        fault_data[i] = nums
        for j in m.tripleList:
            if i == 'F' + str(j.row):
                for k in range(0, data_size):
                    if nums[k] == 0:
                        test_data.loc[k, 'T' + str(j.col)] = 0
        print(i)
    test_data.fillna(1, inplace=True)
    nodes = nodes_test + nodes_fault
    train_data = pd.concat([test_data, fault_data], axis=1)
    # step4: Building a network model
    model = BayesianNetwork()
    for node in nodes:
        model.add_node(node)
    for edge in m.tripleList:
        model.add_edge('T'+str(edge.col), 'F' + str(edge.row))
    # step5: Parameter Learning
    model.fit(train_data)
    # step6: Save the model
    model.save(model_path, filetype='xmlbif')


def infer(model, F, faults, test_dic):
    """
    By using the pre-trained model the test results to get the probability results.

    Args:

    model: the pre-trained bay-net model.

    F: the fault list, and the length of it is equal to fault numbers.

    faults: the faults list contain the real name of each fault.

    test_dic: the test results of each indicator.
    """
    # Network Inference
    model_infer = VariableElimination(model)
    result = {}
    for index, f in enumerate(F[0]):
        prob = model_infer.query(
            variables=[f],
            evidence=test_dic,
        )
        result[faults[index]] = prob.values[0]
    return result


def baynet_inference(D_matrix, model_path, test_result):
    """
    Use the pre-trained model and test_result to get probability.

    Args:
    D_matrix: The original matrix used for training, but in this function only uses its header name
    to generate test list and fault list.

    model_path: the file path where the bay-net `xmlbif` file is located.

    test_result: the test result as the model input, where actually comes from the real test result.
    """
    # Read Bayesian Networks Models
    model = BayesianNetwork.load(model_path, filetype='xmlbif')
    # Read Node
    data = pd.read_excel(D_matrix, sheet_name='Sheet1', index_col=0)
    # Output Node
    faults = data.index.values
    faultNum = 0
    F = []
    for fault in faults:
        F.append('F' + str(faultNum))
        faultNum = faultNum + 1
    F = pd.DataFrame(F)
    # Input node
    tests = data.columns.values
    testNum = 0
    T = []
    for test in tests:
        T.append('T' + str(testNum))
        testNum = testNum + 1
    # Input test results
    test_dic = {}
    flag = 0
    for test in T:
        test_dic[test] = str(test_result[flag])
        flag = flag + 1
    result = infer(model, F, faults, test_dic)
    return result


if __name__ == '__main__':
    print('Please use me as a module!')
