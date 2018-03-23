import numpy as np
import numpy.matlib
from scipy.stats import gamma

#import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""
This file includes required functions for calculating Hillbert-Schmidt Norms. 
"""

def Gaussian_kernel_matrix(darray, sigma = 1):
    """
    This is the function for calculating gram matrix.
    The definition of Kernel between "x" and "y" is as follows; exp{-(1/sigma^2)*(x - y)^2}
    :param darray: The data which we want to calcurate gram matrix.
        The data should be given as follow.
        np.array([
        [data 1]
        [data 2]
        [data 3]
        ...
        [data n]])
        data i can be multi-dimensional, like follows.
        data 1 = [3,1,5,7,8]
    :param sigma: The equation; exp{-(1/sigma^2)*(x - y)^2} requires sigma as a parameter.
    :return: gram matrix
    """
    n_data = np.shape(darray)[0]
    mat_v = np.matlib.repmat(darray, n_data, 1).reshape(n_data, n_data, -1)
    mat_h = np.matlib.repmat(darray, 1, n_data).reshape(n_data, n_data, -1)
    d = mat_v - mat_h
    dd = d * d
    base = - ((1/sigma)**2) *  dd.sum(axis = 2)
    matrix = np.exp(base)
    return matrix

def Laprasian_kernel_matrix(darray, beta = 3):
    """
    This is the function for calculating gram matrix.
    The definition of Kernel between "x" and "y" is as follows; exp{-(1/sigma^2)*x - y)^2}
    :param darray: The data which we want to calcurate gram matrix.
        The data should be given as follow.
        np.array([
        [data 1]
        [data 2]
        [data 3]
        ...
        [data n]])
        data i can be multi-dimensional, like follows.
        data 1 = [3,1,5,7,8]
    :param sigma: The equation; exp{-(1/sigma^2)*x - y)^2} requires sigma as a parameter.
    :return: gram matrix
    """
    n_data = np.shape(darray)[0]
    mat_v = np.matlib.repmat(darray, n_data, 1).reshape(n_data, n_data, -1)
    mat_h = np.matlib.repmat(darray, 1, n_data).reshape(n_data, n_data, -1)
    d = np.absolute(mat_v - mat_h)
    base = - beta *  d.sum(axis = 2)
    matrix = np.exp(base)
    return matrix

def _sigmapi(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    temp = X / Y
    return(np.sum(temp) * np.product(Y))

def _last(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    l = len(X)
    if l <= 2:
        return 0
    else:
        X_hor = X.reshape(l, 1)
        X_ver = X.reshape(1, l)
        mat = np.dot(X_hor, X_ver)
        Y_hor = Y.reshape(l, 1)
        Y_ver = Y.reshape(1, l)
        mat = mat / Y_hor / Y_ver
        return((np.sum(mat) - np.sum(np.diag(mat))) * np.product(Y))



def dHSIC(data):
    d = len(data)
    n = np.shape(data[0])[0]
    term1 = np.ones((n, n))
    term2 = 1
    term3 = 2 / n * np.ones((1, n))
    for i in range(d):
        term1 = term1 * data[i]
        term2 = 1/n/n * term2 * np.sum(data[i])
        term3 = 1/n * term3 * np.sum(data[i], axis = 0, keepdims = True)
    answer = 1/n/n * np.sum(term1) + term2 - np.sum(term3)
    return answer


def dHSIC_gamma_test(data):
    """
    returns p-value of H0; "two datasets are independent"
    :param data:
    :return:
    """
    d = len(data)
    n = np.shape(data[0])[0]
    term1 = np.ones((n, n))
    term2 = 1
    term3 = 2 / n * np.ones((1, n))
    for i in range(d):
        term1 = term1 * data[i]
        term2 = 1 / n / n * term2 * np.sum(data[i])
        term3 = 1 / n * term3 * np.sum(data[i], axis=0, keepdims=True)
    answer = 1 / n / n * np.sum(term1) + term2 - np.sum(term3)
    #Gamma approximation
    #calcurate e
    e_list = []
    for i in range(d):
        e_list.append([1/n/n * np.sum(data[i]), 1/n/n * np.sum(data[i] * data[i]), 1/n/n/n * np.sum(np.sum(data[i], axis = 0) ** 2)])
    e_list = np.array(e_list)
    e_list = e_list.transpose()

    #Calcurate E
    E = (1 - _sigmapi(np.ones(d), e_list[0]) + (d - 1) * np.product(e_list[0])) / n

    #Calcurate Var
    temp = 2
    for k in range(n - 2 * d + 1, n + 1):
        temp = temp / k
    for k in range(n - 4 * d + 3, n - 2 * d + 1):
        temp = temp * k
    #e_list_temp = e_list
    Var = temp * (np.product(e_list[1]) + (d - 1) ** 2 * np.product(e_list[0] ** 2) + 2 * (d - 1) * np.product(e_list[2]) +\
        _sigmapi(e_list[1], e_list[0] ** 2) - 2 * _sigmapi(e_list[1], e_list[2]) - 2 * (d - 1) * _sigmapi(e_list[2], e_list[0] ** 2) + \
        _last(e_list[2], e_list[0] ** 2))
    alpha = E * E / Var
    beta = n * Var / E
    #print(alpha, beta)
    ans = 1 - gamma.cdf(answer * n, alpha, loc=0, scale=beta)
    return ans

if __name__ == "__main__":
    #Data Generation
    m = 500
    data1 = np.random.rand(m, 1)
    data1 = 10 * (data1 - 0.5)
    data2 = data1 ** 2 + np.random.randn(m, 1) * 100

    plt.scatter(data1, data2)
    #plt.show()
    plt.savefig("fig.png")

    #calculate gram matrix
    data1 = Gaussian_kernel_matrix(data1)
    data2 = Gaussian_kernel_matrix(data2)
    #concat
    data = [data1, data2]

    answer = dHSIC_gamma_test(data)
    #print("pval = {}".format(answer))

