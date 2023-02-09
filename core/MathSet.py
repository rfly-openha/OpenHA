# -*- coding: utf-8 -*-
# Copyright (C) rflylab from School of Automation Science and Electrical Engineering, Beihang University.
# All Rights Reserved.


class MathSet:
    '''
    定义该类用于表述数学中集合的概念。
    具体来说，使用表示元素对集合关系的隶属函数来定义集合。
    隶属函数`membership_func()`返回值为`bool`，表示元素是否属于该集合；
    或为0到1之间的浮点数，表示元素对集合的隶属度

    Attributes:
        membership_func: 隶属函数

    '''

    def __init__(self, membership_func: function) -> None:
        '''
        构造函数

        传入一个可调用的函数，作为某集合的隶属函数


        Args:
            membership_func: function, 隶属函数，返回值为bool或0~1之间的float

        '''
        self.membership_func = membership_func

    def has(self, element) -> bool:
        '''
        判断元素`element`对该集合的隶属关系

        Args:
            element: 某一元素或迭代器，类型为隶属函数`membership_func()`可接受的参数类型

        Returns:
            元素`element`对该集合的隶属关系，bool或float
            若`element`为可迭代类型，则返回迭代器

        '''
        # 判断参数element是否刻迭代
        if hasattr(element, '__iter__'):
            # 依次计算每个元素隶属度
            for i in element:
                yield self.membership_func(i)
        else:
            # 直接计算返回该元素的隶属度
            return self.membership_func(element)


def get_interval(
    left: int, right: int, l_closed: bool = False, r_closed: bool = False
) -> MathSet:
    '''
    生成数值区间的集合

    根据输入参数生成一个数值区间的集合。
    集合的左右开闭情况取决于输入参数`l_closed`和`r_closed`。

    Args:
        left: int, 区间的左端点
        right: int, 区间的右端点
        l_closed: bool, optional, 区间的左端是否为闭，默认为`False`
        r_closed: bool, optional, 区间的右端是否为闭，默认为`False`

    Returns:
        类`MathSet`的一个实例，即一个数值区间。
    '''
    if l_closed:
        if r_closed:
            # 闭区间
            f = lambda x: left <= x <= right
        else:
            # 左闭右开
            f = lambda x: left <= x < right
    else:
        if r_closed:
            # 左开右闭
            f = lambda x: left < x <= right
        else:
            # 开区间
            f = lambda x: left < x < right
    # 返回实例化MathSet类
    return MathSet(f)


if __name__ == '__main__':
    print('Please use me as a module!')
