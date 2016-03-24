from utils import *


class NN:
    def __init__(self, architecture, cost, sigmoid, eta):
        # np.random.seed(9)
        # TODO dodat opciju definiranja da li su output klase mutually exclusive? odnosno da li ce se koristit softmax
        # TODO define the use and scope of class variables and class method arguments
        # TODO add asserts
        self.architecture = architecture
        self.eta = eta
        self.depth = len(architecture)
        self.cost = cost
        self.sigmoid = sigmoid
        self.weights = [0.1 * np.random.randn(n + 1, m) for n, m in zip(self.architecture[:], self.architecture[1:])]
        self.z = [np.ones(i) for i in self.architecture]
        self.batch_size = 0

    def train_on_input_batch(self, input_batch, y):
        assert len(input_batch) == len(y)
        gradient = [np.zeros(i.shape) for i in self.weights]

        self.batch_size = len(input_batch)
        self.z = self.forward_pass_batch(input_batch)

        delta = self.calc_delta(y)  # calculates all deltas
        gradient = self.calc_gradient(gradient, delta)  # calculates all gradients
        cost = self.calc_cost(y)

        cost /= len(input_batch)
        gradient = [i / len(input_batch) for i in gradient]

        # self.gradient_check(input_batch, gradient, y)

        self.weights = [item - self.eta * gradient[i] for i, item in enumerate(self.weights)]
        return cost

    def gradient_check(self, input_batch, gradient, y):
        print("----------------------------Gradient check:-----------------------------------------")
        eps = 1e-5
        weights_grad = list(map(np.zeros_like, self.weights))
        for i in range(len(self.weights)):
            xx, yy = self.weights[i].shape
            for j in range(xx):
                for k in range(yy):
                    med = self.weights[i][j, k]
                    self.weights[i][j, k] = med + eps
                    z_uppercost = self.forward_pass_batch(input_batch)
                    self.weights[i][j, k] = med - eps
                    z_lowercost = self.forward_pass_batch(input_batch)
                    self.weights[i][j, k] = med

                    cost_upper = np.sum(self.cost.f(z_uppercost[-1], y))
                    cost_lower = np.sum(self.cost.f(z_lowercost[-1], y))
                    weights_grad[i][j, k] = (cost_upper - cost_lower) / (2 * eps * len(input_batch))
        for i in self.weights:
            print(i)

        print("numerical estimation:")
        for i in weights_grad:
            print(i)

        print("backprop:")
        for i in gradient:
            print(i)

        print("Difference:")
        for i in range(len(gradient)):
            print(np.absolute((weights_grad[i] - gradient[i])) / (
                np.maximum(np.absolute(weights_grad[i]), np.absolute(gradient[i]))))
        print("")
        # input()

    def calc_cost(self, y):
        return np.sum([i for i in self.cost.f(self.z[-1], y)])

    def calc_delta(self, y, **kwargs):
        w = kwargs.get("w", self.weights)
        z = kwargs.get("z", self.z)
        # TODO change delta to its more complex form:
        delta = [0 for _ in range(self.depth)]
        delta[-1] = self.z[-1] - y
        for l in reversed(range(1, self.depth - 1)):  # Update deltas in the reverse order
            delta[l] = np.dot(delta[l + 1], w[l].T)[:, :-1] * self.sigmoid.df(-np.log(1 / z[l] - 1))
        return delta

    def calc_gradient(self, gradient, delta):
        promjena = list()
        for layer in range(self.depth - 1):
            promjena.append(np.sum([
                                       np.dot(np.concatenate([self.z[layer][j], [1]])[np.newaxis].T,
                                              delta[layer + 1][j][np.newaxis])
                                       for j in range(self.batch_size)
                                       ], axis=0))
        for i in range(len(gradient)):
            gradient[i] += promjena[i]

        return gradient

    def forward_pass_batch(self, input_batch, **kwargs):
        """
        :param input_batch: 2d array in which the rows represent individual input examples
        :param w: optional weight parameter, if not passed, the function uses the class instance's weights
        :return: returns the output layer
        """
        w = kwargs.get("w", self.weights)
        n = len(input_batch)
        z = [input_batch]
        for i in range(self.depth - 1):  # last layer is softmax
            z.append(self.sigmoid.f(np.dot(np.c_[z[-1], np.ones(n)], w[i])))
        # z.append(SoftMax.f(np.dot(np.c_[z[-1], np.ones(n)], w[-1])))
        return z
