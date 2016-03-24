import itertools
import math
import random
import pickle

from plot import *


class Mnist:
    def __init__(self):
        f = open("mnist/mnist.pkl")
        train_set, valid_set, test_set = pickle.load(f)
        f.close()


class Line1d:
    low = 0
    upp = 1
    draw = 0
    dim_samples = 50
    resolution = 10
    n_graphs = 1

    def __init__(self):
        self.tr_set = Line1d.generate_examples(5)
        print(self.tr_set)
        self.x = np.arange(self.low, self.upp, np.abs(self.upp - self.low) / self.dim_samples)
        self.y1 = [int(Line1d.cf_middle(i)) for i in self.x]
        self.y2 = [int(Line1d.cf_outer(i)) for i in self.x]
        self.graf = Plot(self.x, [0, 0.5, 1], plot_3d=False, plot_cost=False, res=self.resolution,
                         n_classes=self.n_graphs, lower=self.low, upper=self.upp)

        # duplicating data along the y axis so we can draw it
        self.z = np.expand_dims(self.tr_set, axis=2).repeat(3, 2).T
        # print(self.z.shape)
        # print(self.z)
        values = [np.expand_dims(i, axis=1).repeat(3, 1).T for i in [self.y1, self.y2]]
        print(values[0].shape)
        self.graf.draw(values)

    @staticmethod
    def generate_examples(n):
        koja_funkcija = np.random.rand(n, 1)
        return np.array([np.array([Line1d.f_middle(), 1]) if item > 0.5 else np.array(
            [Line1d.f_outer(), 0]) for item in koja_funkcija])

    @staticmethod
    def cf_outer(x):
        return x < 0.2 or 0.8 <= x

    @staticmethod
    def cf_middle(x):
        return 0.4 <= x < 0.6

    @staticmethod
    def f_middle():
        return random.random() / 5 + 0.4

    @staticmethod
    def f_outer():
        return random.random() / 5 + np.random.randint(2) * 0.8


class TwoLines:
    low = -1
    upp = 1
    draw = 0
    dim_samples = 50
    resolution = 10
    n_graphs = 1

    def __init__(self, size, draw=1, n_graphs=1, resolution=10, dimension_samples=50):
        """

        :param size: size of the train set
        :param draw:
        :param n_graphs:
        :param resolution: resolution of the graph
        :param dimension_samples: number of points sampled on a dimension in the graph
        :return:
        """
        self.draw = draw
        self.resolution = resolution
        self.dim_samples = dimension_samples
        self.n_graphs = n_graphs
        self.tr_set = np.array([TwoLines.generate_example() for _ in range(size)])
        self.trset_size = size
        self.x, self.y = [np.arange(self.low, self.upp, np.abs(self.upp - self.low) / self.dim_samples)] * 2
        self.all_combinations = np.array([list(x) for x in itertools.product(self.x, self.y)])
        if self.draw:
            self.graf = Plot(self.x, self.y, plot_3d=False, plot_cost=True, res=self.resolution,
                             n_classes=self.n_graphs)

        self.cost_low = -15
        self.cost_upp = 15
        self.x_c, self.y_c = [np.arange(self.cost_low, self.cost_upp,
                                        np.abs(self.cost_upp - self.cost_low) / self.dim_samples)] * 2
        self.cost_3d = Plot(self.x_c, self.y_c, plot_3d=True, res=self.resolution)

    # def plot_cost(self):

    def plot(self, layer, cost):
        """

        :param layer: 2d array to be visualised
        :param cost: a list of costs over epochs
        """
        values = layer.reshape([self.dim_samples, self.dim_samples, 2], order="F")
        brojevi = np.arange(self.low, self.upp, 0.01)
        c = cost[5:] if len(cost) > 6 else cost
        self.graf.draw([values[:, :, i] for i in range(2)], c)
        for i in range(self.n_graphs):
            # self.graf.ax_c[i].plot(brojevi, [np.sqrt((0.25 - i ** 2)) for i in brojevi])
            # self.graf.ax_c[i].plot(brojevi, [-np.sqrt((0.25 - i ** 2)) for i in brojevi])

            self.graf.ax_c[i].plot(brojevi, [TwoLines.f_bottom(i) for i in brojevi])
            self.graf.ax_c[i].plot(brojevi, [TwoLines.f_top(i) for i in brojevi])

    @staticmethod
    def generate_examplev2():
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1

        if x ** 2 + y ** 2 <= 0.25:
            return [np.array([x, y]), np.array([1, 0])]
        else:
            return [np.array([x, y]), np.array([0, 1])]

    @staticmethod
    def generate_example():
        koja_funkcija = random.random()
        x = random.random() * 2 - 1
        if koja_funkcija > 0.5:
            return [np.array([x, TwoLines.f_bottom(x)]), np.array([1, 0])]
        else:
            return [np.array([x, TwoLines.f_top(x)]), np.array([0, 1])]

    @staticmethod
    def f_bottom(x):
        return math.cos(x * 3) / 2 - 0.3  # + 17**(x - 0.7 + abs(x - 0.7)) - 1

    @staticmethod
    def f_top(x):
        return math.cos(x * 3) / 2 + 0.3  # + 17**(x - 0.5 + abs(x - 0.5)) - 1 #,, 2**(x*2) - 2


"""
        graf[0].add_slider([i.shape for i in nn.weights], -5, 5, 0)


        def update(val):
            slider_value = [[j.val for j in layer] for layer in graf[0].slider]
            print([i for i in nn.weights], "\n --------------------")
            for i in nn.weights:
                print(i)
            for layer_number, layer in enumerate(slider_value):
                for i, item in enumerate(layer):
                    # print([int(i/nn.weights[0].shape[1]), i%nn.weights[0].shape[1]])
                    nn.weights[layer_number][
                        int(i / nn.weights[layer_number].shape[1]), i % nn.weights[layer_number].shape[1]] = item

            all_comb_predicted = nn.forward_pass_batch(all_combinations)[-1]
            all_comb_image = all_comb_predicted.reshape([dimension_samples, dimension_samples, 2], order="F")
            for j in range(len(graf)):
                graf[j].draw(all_comb_image[:, :, j])


        for i in graf[0].slider:
            for j in i:
                j.on_changed(update)

"""
