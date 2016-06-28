import itertools
import numpy as np
from plot import *


class ConsoleOutput:
    @staticmethod
    def update(*args):
        nn, i, cijena, data_feed = args[:4]
        print("Batch #", i, "  Last cost: ", cijena, sep="")


class NetworkOutput:
    low = -1
    upp = 1
    draw = 0
    dim_samples = 60
    resolution = 20
    n_graphs = 1

    def __init__(self, draw=1, n_graphs=1, resolution=20, dimension_samples=60, plot_cost=True, plot_3d=False,
                 plot_cost_hyp=False):
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
        self.x, self.y = [np.arange(self.low, self.upp, np.abs(self.upp - self.low) / self.dim_samples)] * 2
        self.all_combinations = np.array([list(x) for x in itertools.product(self.x, self.y)])
        if self.draw:
            self.graf = Plot(self.x, self.y, plot_3d=plot_3d, plot_cost=plot_cost, res=self.resolution,
                             plot_cost_hyp=plot_cost_hyp,
                             n_classes=self.n_graphs)

        self.cost_low = -15
        self.cost_upp = 15
        self.x_c, self.y_c = [np.arange(self.cost_low, self.cost_upp,
                                        np.abs(self.cost_upp - self.cost_low) / self.dim_samples)] * 2
        # self.cost_3d = Plot(self.x_c, self.y_c, plot_3d=True, res=self.resolution)
        self.cost_list = []

    def update(self, *args):
        nn, i, costt, data_feed = args[:4]
        self.cost_list.append(costt)

        nn.forward_pass_batch(self.all_combinations)
        hidden_layer, output_layer = nn.a[-2:]
        output_values = output_layer.reshape([self.dim_samples, self.dim_samples, 2], order="F")
        # hidden_values = hidden_layer.reshape([self.dim_samples, self.dim_samples, 3], order="F")
        c = self.cost_list[5:] if len(self.cost_list) > 6 else self.cost_list
        graph_list = [output_values[:, :, i] for i in range(self.n_graphs)]
        # graph_list.extend([hidden_values[:, :, i] for i in range(3)])
        self.graf.draw(graph_list, c)

        samples = np.arange(self.low, self.upp, 0.0001)
        for i in range(self.n_graphs):
            for f in data_feed.get_functions():
                self.graf.ax_c[i].plot(samples, list(map(f, samples)), "b")
