import math
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import random
import time

lower = -1
upper = 1


def fun0(x):
    return math.cos(x*3)/2 - 0.4  #+ 17**(x - 0.7 + abs(x - 0.7)) - 1


def fun1(x):
    return math.cos(x*3)/2 + 0.4  #+ 17**(x - 0.5 + abs(x - 0.5)) - 1,, 2**(x*2) - 2


def generate_example():
    k = random.random()
    r = random.random() * 2 - 1
    if k > 0.5:
        return [np.array([r, fun0(r)]), np.array([1, 0]), np.array([0, 0])]
    else:
        return [np.array([r, fun1(r)]), np.array([0, 1]), np.array([0, 0])]


class Sigmoid:

    @staticmethod
    def f(z):
        try:
            return [1/(1 + math.exp(-i)) for i in z]
        except OverflowError:
            print(z)

    @staticmethod
    def df(z):
        return [i*(1 - i) for i in z]


class SoftMax:

    @staticmethod
    def f(y):
        total = np.sum([math.exp(i) for i in y])
        return [math.exp(i)/total for i in y]

    @staticmethod
    def df(y):
        return [i/(1 - i) for i in y]


class CrossEntropy:

    @staticmethod
    def f(a, y):
        assert y.shape == a.shape
        return np.sum([-(y[i]*math.log(a[i]) + (1 - y[i])*math.log(1 - a[i])) for i in range(len(a))])

    @staticmethod
    def df(a, y):
        assert y.shape == a.shape
        return a - y


class SqrCost:

    @staticmethod
    def f(a, y):
        assert y.shape == a.shape
        return np.square(a - y)/2

    @staticmethod
    def df(a, y):
        assert y.shape == a.shape
        return a - y


class NN:

    inputlayer_size = 2
    hiddenlayer_size = 0
    outputlayer_size = 2
    eta = 50

    def __init__(self, cost, sigmoid):
        #np.random.seed(7)
        self.cost = cost
        self.sigmoid = sigmoid
        self.weights = np.random.rand(self.inputlayer_size + 1, self.outputlayer_size)*2 - 1

    def process_input_batch(self, input_batch, y):
        cost = 0
        gradient = np.zeros(self.weights.shape)
        for i, item in enumerate(input_batch):
            input_layer = np.concatenate([item, np.array([1])])
            output_layer = np.array(self.forward_pass(input_layer, self.weights))

            delta = np.array(self.cost.df(output_layer, y[i])*self.sigmoid.df(output_layer))
            gradient += np.dot(np.transpose(input_layer[np.newaxis]), delta[np.newaxis])
            cost += self.cost.f(output_layer, y[i])

        cost /= len(input_batch)
        gradient /= len(input_batch)

        self.weights -= self.eta*gradient
        return cost

    def forward_pass(self, layer1, weights):
        return SoftMax.f(self.sigmoid.f(np.dot(layer1, weights)))


class PlotContour:


    def __init__(self):
        matplotlib.interactive(True)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.X = np.arange(-1, 1, 0.01)
        self.Y = np.arange(-1, 1, 0.01)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        self.Z = self.X + self.Y
        self.ax.contourf(self.X, self.Y, self.Z, cmap=cm.afmhot, nchunk = 0)
        plt.ion()

    def draw(self, tezine):
        self.Z = self.X*tezine[0] + self.Y * tezine[1]
        self.ax.contourf(self.X, self.Y, self.Z, cmap=cm.afmhot, nchunk = 0)

        brojevi = np.arange(lower, upper, 0.01)

        self.ax.plot(brojevi, [fun0(i) for i in brojevi])
        self.ax.plot(brojevi, [fun1(i) for i in brojevi])
        #self.ax.axes().set_aspect('equal')
        self.ax.axis((lower, upper, lower, upper))

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
        self.Z = self.X*tezine[0] + self.Y * tezine[1]
        self.surf.remove()
        self.surf = self.ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                         linewidth=0, antialiased=False)
        plt.draw()


nn = NN(CrossEntropy, Sigmoid)
graf = PlotContour()
graf2 = PlotContour()

training_set_size = 30000
batch_size = 1000
training_set = np.array([generate_example() for i in range(training_set_size)])


cost = list()
ukupno = list()
u = 0
for i in [training_set[j:j + batch_size] for j in range(0, training_set_size, batch_size)]:
    cijena = nn.process_input_batch(i[:, 0], i[:, 1])
    cost.append(cijena)
    graf.draw(nn.weights[:, 0])
    graf2.draw(nn.weights[:, 1])
    print(u)
    u += 1


matplotlib.interactive(False)
plt.figure()
plt.plot(cost)
plt.axis((0, training_set_size/batch_size, 0, 2))
plt.show()

tocno = 0
for i in training_set:
    a = np.concatenate([i[0], np.array([1])])
    pred = nn.forward_pass(a, nn.weights)
    i[2] = pred
    if np.argmax(pred) == np.argmax(i[1]):
        tocno += 1
    else:
        print("Input:", i[0], " Output:", pred)
        print("Solution:          ", i[1])

print("Postotak:", tocno/len(training_set))





plt.figure()
brojevi = np.arange(lower, upper, 0.01)

plt.plot(brojevi, [fun0(i) for i in brojevi])
plt.plot(brojevi, [fun1(i) for i in brojevi])
plt.axes().set_aspect('equal')
plt.axis((lower, upper, lower, upper))


for i in training_set:
    if np.argmax(i[1]) == np.argmax(i[2]):
        pass
        #plt.scatter(i[0, 0], i[0, 1], c=1)
    else:
        plt.scatter(i[0, 0], i[0, 1], c=6)

plt.show()
