import itertools
import math
import random
import pickle

from plot import *


class Data:
    def __init__(self, data_feed):
        self.data_feed = data_feed

    def next_batch(self, n_batches, batch_size):
        for _ in range(n_batches):
            yield np.array([self.data_feed.generate_example() for _ in range(batch_size)]).transpose((1, 0, 2))


class DataGenerator:
    @staticmethod
    def generate_example():
        raise NotImplementedError()

    @staticmethod
    def get_functions():
        raise NotImplementedError()


class Circle(DataGenerator):
    @staticmethod
    def generate_example():
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1

        if x ** 2 + y ** 2 <= 0.25:
            return [np.array([x, y]), np.array([1, 0])]
        else:
            return [np.array([x, y]), np.array([0, 1])]

    @staticmethod
    def get_functions():
        return [Circle.half_circle, Circle.neg_half_circle]

    @staticmethod
    def half_circle(i):
        return np.sqrt((0.25 - i ** 2))

    @staticmethod
    def neg_half_circle(i):
        return -Circle.half_circle(i)


class TwoLines(DataGenerator):
    @staticmethod
    def generate_example():
        koja_funkcija = random.random()
        x = random.random() * 2 - 1
        if koja_funkcija > 0.5:
            return [np.array([x, TwoLines.f_bottom(x)]), np.array([1, 0])]
        else:
            return np.array([x, TwoLines.f_top(x)]), np.array([0, 1])

    @staticmethod
    def get_functions():
        return [TwoLines.f_bottom, TwoLines.f_top]

    @staticmethod
    def f_bottom(x):
        return math.cos(x * 3) / 2 - 0.3  # + 17**(x - 0.7 + abs(x - 0.7)) - 1

    @staticmethod
    def f_top(x):
        return math.cos(x * 3) / 2 + 0.3  # + 17**(x - 0.5 + abs(x - 0.5)) - 1 #,, 2**(x*2) - 2
