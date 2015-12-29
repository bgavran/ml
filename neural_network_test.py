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
    return math.cos(x*3)/2 - 0.4


def fun1(x):
    return math.cos(x*3)/2 + 0.4


def generate_example():
    k = random.random()
    r = random.random() * 2 - 1
    if k > 0.5:
        return np.array([r, fun0(r), 0])
    else:
        return np.array([r, fun1(r), 1])


class Sigmoid:

    @staticmethod
    def f(z):
        try:
            return 1/(1 + math.exp(-z))
        except OverflowError:
            print(z)

    @staticmethod
    def df(z):
        return z*(1 - z)


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
    outputlayer_size = 1
    eta = 0.001

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
            gradient += np.dot(delta, self.weights)
            cost += self.cost.f(output_layer, y[i])

        gradient *= self.eta
        cost /= len(input_batch)
        gradient /= len(input_batch)

        self.weights -= gradient
        return cost

    def forward_pass(self, layer1, weights):
        return self.sigmoid.f(np.dot(layer1, weights))


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

        

    

nn = NN(SqrCost, Sigmoid)
graf = Plot3D()

number_of_batches = 100
batch_size = 10
cost = list()
ukupno = list()
for i in range(number_of_batches):
    examples = np.empty((batch_size, 3))
    for j in range(batch_size):
        primjer = generate_example()
        examples[j] = primjer
        ukupno.append(primjer)
    cijena = nn.process_input_batch(examples[:, :2], examples[:, 2])
    cost.append(cijena)
    #graf.draw(nn.weights)

    if i % 10 == 0:
        print(i, nn.weights, )

matplotlib.interactive(False)
plt.figure(2)
plt.plot(cost)
plt.axis((0, number_of_batches, 0, 0.5))
plt.show()

koliko = 1000
tocno = 0
for i in range(koliko):
    t = generate_example()
    a = np.concatenate([t[:2], np.array([1])])
    pred = nn.forward_pass(a, nn.weights)
    rj = 1 if pred >= 0.5 else 0
    if t[2] == rj:
        tocno += 1

print("Postotak:", tocno/koliko)





plt.figure(3)
brojevi = np.arange(lower, upper, 0.01)

plt.plot(brojevi, [fun0(i) for i in brojevi])
plt.plot(brojevi, [fun1(i) for i in brojevi])
plt.axes().set_aspect('equal')
plt.axis((lower, upper, lower, upper))

for i in ukupno:
    if i[2] == 0:
        plt.scatter(i[0], i[1], c=1)
    else:
        plt.scatter(i[0], i[1], c=6)

plt.show()
