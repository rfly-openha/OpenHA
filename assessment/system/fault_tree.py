# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


def gate_or(q1: float, q2: float) -> float:
    '''
    The boolean logic gate `or` in the fault tree.
    Calculate the probability that the conditioning event happens according to the inputs.

    Args:
        q1: float. The probability that an basic event happens.
        q2: float. The probability that the other basic event happens.

    '''
    return 1 - (1 - q1) * (1 - q2)


def gate_and(q1: float, q2: float) -> float:
    '''
    The boolean logic gate `and` in the fault tree.
    Calculate the probability that the conditioning event happens according to the inputs.

    Args:
        q1: float. The probability that an basic event happens.
        q2: float. The probability that the other basic event happens.

    '''
    return q1 * q2


def gate_not(q1: float) -> float:
    '''
    The boolean logic gate `not` in the fault tree.
    Calculate the probability that the conditioning event happens according to the inputs.

    Args:
        q1: float. The probability that an basic event happens.

    '''
    return 1 - q1


def gate_xor(q1: float, q2: float) -> float:
    '''
    The boolean logic gate `xor` in the fault tree.
    Calculate the probability that the conditioning event happens according to the inputs.

    Args:
        q1: float. The probability that an basic event happens.
        q2: float. The probability that the other basic event happens.

    '''
    return (1 - q1) * q2 + q1 * (1 - q2)
