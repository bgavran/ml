import numpy as np
import math
import random
from decimal import Decimal


def fun0(x):
    return math.cos(x*3)/2 - 0.3  # + 17**(x - 0.7 + abs(x - 0.7)) - 1


def fun1(x):
    return math.cos(x*3)/2 + 0.3  # + 17**(x - 0.5 + abs(x - 0.5)) - 1 #,, 2**(x*2) - 2


def generate_examplev2():
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    if x**2 + y**2 <= 0.25:
        return[np.array([x, y]), np.array([1, 0])]
    else:
        return[np.array([x, y]), np.array([0, 1])]


def generate_example():
    koja_funkcija = random.random()
    x = random.random() * 2 - 1
    if koja_funkcija > 0.5:
        return [np.array([x, fun0(x)]), np.array([1, 0])]
    else:
        return [np.array([x, fun1(x)]), np.array([0, 1])]


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



