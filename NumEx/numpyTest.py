import numpy as np
from numpy.linalg import *


def main1():
    # lst = [[1, 3, 5], [2, 4, 6], [4, 6, 8]]
    # np_lst = np.array(lst)
    # print(np_lst)
    # print(np_lst.shape)
    # print(np_lst.ndim)
    # print(np_lst.dtype)
    # print(np_lst.itemsize)
    # print(np_lst.size)
    #
    # print(np.zeros([2, 4]))
    # print(np.ones([3, 5]))
    # print("Rand")
    # print(np.random.rand(2, 4))
    # print(np.random.rand())
    # print("RandInt")
    # print(np.random.randint(1, 10))
    # print(np.random.randn(2, 4))
    # print(np.random.choice([10, 20, 30]))

    # print("Distribute:")
    # print(np.random.beta(1, 10, 100))

    # lst = np.arange(1, 11).reshape(2, -1)
    # print("exp")
    # print(np.exp(lst))
    # print("exp2")
    # print(np.exp2(lst))
    # print("sqrt")
    # print(np.sqrt(lst))
    # print("sin")
    # print(np.sin(lst))
    # print("log")
    # print(np.log(lst))

    # lst = np.array([[[1, 2, 3, 4], [4, 5, 6, 7]],
    #                 [[7, 8, 9, 10], [10, 11, 12, 13]],
    #                 [[14, 15, 16, 17], [18, 19, 20, 21]]
    #                 ])
    # print(lst.shape)
    # print(lst.ndim)
    # print(lst.sum(axis=0))
    # print(lst.sum(axis=1))
    # print(lst.sum(axis=2))
    #
    # print(lst.max(axis=1))
    # print(lst.min(axis=1))

    # lst1 = np.array([10, 20, 30, 40])
    # lst2 = np.array([4, 3, 2, 1])
    # print(lst1 + lst2)
    # print(lst1 - lst2)
    # print(lst1 * lst2)
    # print(lst1 / lst2)
    # print(lst1 ** 2)
    # print(np.dot(lst1, lst2))
    # print(lst1.reshape([2, 2]))
    # print(np.dot(lst1.reshape([2, 2]), lst2.reshape([2, 2])))

    # print(np.concatenate((lst1, lst2), axis=0))
    # print(np.concatenate((lst1.reshape([2, 2]), lst2.reshape([2, 2])), axis=1))
    # print(np.vstack((lst1, lst2)))
    # print(np.hstack((lst1, lst2)))
    # print(np.split(lst1, 4))
    # print(np.copy(lst1))

    # 4 liner

    print(np.eye(3))  # 对角矩阵
    lst = np.array([[1., 2.],
                    [3., 4.]])
    print(lst.dtype)
    print(inv(lst))
    print(lst.transpose())
    print(det(lst))
    print(eig(lst))
    y = np.array([[5.], [7.]])
    print(solve(lst, y))


if __name__ == '__main__':
    main1()
