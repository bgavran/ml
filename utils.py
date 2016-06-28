import copy
import numpy as np


class Sigmoid:

    @staticmethod
    def f(z):
        return 1/(1 + np.exp(-z))
        # return [1/(1 + math.exp(-i)) for i in z]

    @staticmethod
    def df(z):
        return Sigmoid.f(z)*(1 - Sigmoid.f(z))


class SoftMax:

    @staticmethod
    def f(z):
        # Calculates the sum of exp(elemenets) of each row of the array. Divides each element by the corresponding sum
        total = np.sum(np.exp(z), axis=1)
        return np.exp(z)/total[:, np.newaxis]

    @staticmethod
    def df(z):
        return SoftMax.f(z)*(1 - SoftMax.f(z))


class CrossEntropyCost:

    @staticmethod
    def f(a, y):
        assert y.shape == a.shape
        return -np.sum(np.nan_to_num(y*np.log(a) + (1 - y)*np.log(1 - a)), axis=1)

    @staticmethod
    def df(a, y):
        assert y.shape == a.shape
        return (a - y)/(a*(1 - a))


class SqrCost:

    @staticmethod
    def f(a, y):
        assert y.shape == a.shape
        return np.square(a - y)/2

    @staticmethod
    def df(a, y):
        assert y.shape == a.shape
        return a - y



