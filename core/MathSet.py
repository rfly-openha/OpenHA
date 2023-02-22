# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


class MathSet:
    '''
    This class is the abstract of Set in mathematics.
    The membership function `membership_func()` returns a value of `bool`, indicating whether the element belongs to the set.
    Or a float number between 0 and 1, indicating the element's degree of membership to the set

    Attributes:
        membership_func: membership funcion that indicate the relationship between an element and a set.

    '''

    def __init__(self, membership_func) -> None:
        '''
        Constructed Function

        Take a function that can be called as a membership function of a set.

        Args:
            membership_func: the membership funciton.

        '''
        self.membership_func = membership_func

    def has(self, element) -> bool | list[bool]:
        '''
        Tell the relationship between `element` and this set.

        Args:
            element: An element or iterator, acceptable to the  `membership_func()`.

        Returns:
            bool or float, can be list if `element` is an array.

        '''
        # tell whether `element` is iterable
        if hasattr(element, '__iter__'):
            # compute the degree of membership of each
            return [self.membership_func(i) for i in element]
        else:
            # compute the degree of membership
            return self.membership_func(element)


def get_interval(
    left: int, right: int, l_closed: bool = False, r_closed: bool = False
) -> MathSet:
    '''
    Generate an interval

    Generates an interval based on the arguments.
    The opening and closing of the set depends on `l_closed` and `r_closed`.

    Args:
        left: int, the left endpoint of the interval
        right: int, the right endpoint of the interval
        l_closed: bool, optional, is the left end of the interval closed, default as `False`
        r_closed: bool, optional, Is the right end of the interval closed, default as `False`

    Returns:
        An instant of the class `MathSet`
    '''
    if l_closed:
        if r_closed:
            # closed interval
            f = lambda x: left <= x <= right
        else:
            # left closed, right open
            f = lambda x: left <= x < right
    else:
        if r_closed:
            # open left and close right
            f = lambda x: left < x <= right
        else:
            # open interval
            f = lambda x: left < x < right
    # an instant of the class `MathSet` initialized by the function `f`
    return MathSet(f)


if __name__ == '__main__':
    print('Please use me as a module!')
