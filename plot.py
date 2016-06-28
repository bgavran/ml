import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
import seaborn

from utils import *


class Plot:
    low = -1
    upp = 1

    def __init__(self, x, y, res=10, plot_3d=False, plot_cost=False, plot_cost_hyp=False, n_classes=1, lower=-1,
                 upper=1, interactive=False):
        self.low = lower
        self.upp = upper
        self.plot_3d = plot_3d
        self.plot_cost = plot_cost
        self.plot_cost_hyp = plot_cost_hyp
        self.n_classes = n_classes
        self.resolution = res
        self.fig = plt.figure()
        self.ax_c = []
        self.cost = []
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.X + self.Y
        self.slider = []
        self.drawn_once = 0
        self.surf = []
        matplotlib.interactive(interactive)

        n_plot_rows = 1 + self.plot_cost + self.plot_cost_hyp
        kw = {"aspect": "equal"}
        if self.plot_3d:
            kw["projection"] = "3d"
            for _ in range(self.n_classes):
                self.surf.append(0)
        for i in range(self.n_classes + self.plot_cost_hyp):
            self.ax_c.append(self.fig.add_subplot(n_plot_rows, self.n_classes, i + 1, **kw))

        plt.ion()

    def add_slider(self, shape, minn, maxx, default):
        for layer_number, layer in enumerate(shape):
            self.slider.append(list())
            for i in range(np.prod(layer)):
                self.slider[-1].append(
                    Slider(plt.axes([0.25, 0.05 + 0.03 * np.sum([len(z) for z in self.slider]), 0.65, 0.02],
                                    axisbg="lightgoldenrodyellow"),
                           str(layer_number) + str(". layer:    ") + str(int(i / layer[1])) + "," + str(
                               int(i % layer[1])), minn, maxx,
                           valinit=default))

    def draw(self, z, cost_data=0, ):
        assert cost_data != 0 or not self.plot_cost

        matplotlib.interactive(True)
        if self.drawn_once:
            for i in self.surf:
                i.remove()

        for i in range(self.n_classes):
            if self.plot_3d:
                self.surf[i] = self.ax_c[i].plot_surface(self.X, self.Y, z[i], rstride=1,
                                                         cstride=1,
                                                         cmap=cm.coolwarm,
                                                         linewidth=0, antialiased=False)
                self.drawn_once = 1
            else:
                self.ax_c[i].contourf(self.X, self.Y, z[i], self.resolution, cmap=cm.afmhot, nchunk=0)

            self.ax_c[i].axis((self.low, self.upp, self.low, self.upp))

        if self.plot_cost:
            self.cost = self.fig.add_subplot(3, 1, 3)
            self.cost.plot(cost_data)
            self.cost.axis((0, len(cost_data), 0, max(cost_data)))

        self.fig.canvas.draw()
        plt.pause(0.00001)
