import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.widgets import Slider
from utils import *


class PlotContour:
    lower = -1
    upper = 1

    def __init__(self, x, y, res, interactive=False):
        matplotlib.interactive(interactive)
        self.resolution = res
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.25, bottom=0.2)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.X = x
        self.Y = y
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        self.Z = self.X + self.Y
        self.ax.contourf(self.X, self.Y, self.Z, self.resolution, cmap=cm.afmhot, nchunk=0)
        self.slider = list()
        plt.ion()

    def add_slider(self, shape, minn, maxx, default):
        for layer_number, layer in enumerate(shape):
            self.slider.append(list())
            for i in range(np.prod(layer)):
                self.slider[-1].append(
                    Slider(plt.axes([0.25, 0.05 + 0.03 * np.sum([len(z) for z in self.slider]), 0.65, 0.02], axisbg="lightgoldenrodyellow"),
                           str(layer_number) + str(". layer:    ") + str(int(i / layer[1])) + "," + str(int(i % layer[1])), minn, maxx,
                           valinit=default))

    def draw(self, Z):
        matplotlib.interactive(True)

        # self.Z = self.X * tezine[0] + self.Y * tezine[1]
        self.ax.contourf(self.X, self.Y, Z, self.resolution, cmap=cm.afmhot, nchunk=0)

        brojevi = np.arange(self.lower, self.upper, 0.01)

        self.ax.plot(brojevi, [fun0(i) for i in brojevi])
        self.ax.plot(brojevi, [fun1(i) for i in brojevi])
        # self.ax.axes().set_aspect('equal')
        self.ax.axis((self.lower, self.upper, self.lower, self.upper))

        self.fig.canvas.draw()


class Plot3D:
    def __init__(self):
        matplotlib.interactive(True)
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.X = np.arange(-1, 1, 0.05)
        self.Y = np.arange(-1, 1, 0.05)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        self.Z = 1
        self.surf = self.ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                         linewidth=0, antialiased=False)

        self.ax.set_zlim(-2, 2)

    def draw(self, tezine):
        self.Z = self.X * tezine[0] + self.Y * tezine[1]
        self.surf.remove()
        self.surf = self.ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                         linewidth=0, antialiased=False)
        plt.draw()
