from utils import *


class NN:
    def __init__(self, architecture, cost, sigmoid, eta, observers=()):
        # np.random.seed(9)

        self._observers = observers
        self.architecture = architecture
        self.eta = eta
        self.depth = len(architecture)
        self.cost = cost
        self.activation = sigmoid
        self.weights = [(np.random.rand(n + 1, m) - 0.5) / np.sqrt(n) for n, m in
                        zip(self.architecture[:], self.architecture[1:])]
        # self.bias = [(np.random.rand(1, n) - 0.5)/ np.sqrt(n) for n in self.architecture[:-1]]
        self.z = [None for _ in self.architecture]
        self.a = [None for _ in self.architecture]
        self.batch_size = 0

    def notify(self, *args):
        for obs in self._observers:
            obs.update(self, *args)

    def train_on_input_batch(self, input_batch, y):
        assert len(input_batch) == len(y)

        self.batch_size = len(input_batch)
        self.forward_pass_batch(input_batch)

        delta = self.calc_delta(y)  # calculates all deltas
        gradient = self.calc_gradient(delta)  # calculates all gradients
        cost = self.calc_cost(y)

        # self.gradient_check(input_batch, gradient, y)

        for w, g in zip(self.weights, gradient):
            w -= self.eta * g
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
                    outputs_upper = copy.deepcopy(self.forward_pass_batch(input_batch))
                    self.weights[i][j, k] = med - eps
                    outputs_lower = copy.deepcopy(self.forward_pass_batch(input_batch))
                    self.weights[i][j, k] = med

                    cost_upper = np.sum(self.cost.f(outputs_upper[-1], y))
                    cost_lower = np.sum(self.cost.f(outputs_lower[-1], y))
                    weights_grad[i][j, k] = (cost_upper - cost_lower) / (2 * eps * len(input_batch))

        # for i in self.weights:
        #     print(i)
        #
        # print("numerical estimation:")
        # for i in weights_grad:
        #     print(i)
        #
        # print("backprop:")
        # for i in gradient:
        #     print(i)

        print("Difference:")
        for i in range(len(gradient)):
            print(np.absolute((weights_grad[i] - gradient[i])) / (
                np.maximum(np.absolute(weights_grad[i]), np.absolute(gradient[i]))))
        print("")
        # input()

    def calc_cost(self, y):
        return np.mean(self.cost.f(self.a[-1], y))

    def calc_delta(self, y, **kwargs):
        w = kwargs.get("w", self.weights)
        z = kwargs.get("z", self.z)
        delta = [None for _ in range(self.depth)]
        delta[-1] = self.cost.df(self.a[-1], y) * self.activation.df(z[-1])
        for l in reversed(range(1, self.depth - 1)):  # Update deltas in the reverse order
            delta[l] = (delta[l + 1] @ w[l].T)[:, :-1] * self.activation.df(z[l])
        return delta

    def calc_gradient(self, delta):
        gradient = []
        for layer in range(self.depth - 1):
            gradient.append(np.mean([
                                        (np.concatenate([self.a[layer][j], [1]])[np.newaxis].T @
                                         delta[layer + 1][j][np.newaxis])
                                        for j in range(self.batch_size)
                                        ], axis=0))

        return gradient

    def forward_pass_batch(self, input_batch, **kwargs):
        """
        :param input_batch: 2d array in which the rows represent individual input examples
        :param w: optional weight parameter, if not passed, the function uses the class instance's weights
        :return: returns the output layer
        """
        w = kwargs.get("w", self.weights)
        n = len(input_batch)
        self.a[0] = self.z[0] = input_batch
        for i in range(self.depth - 1):  # last layer is softmax
            self.z[i + 1] = np.c_[self.a[i], np.ones(n)] @ w[i]
            self.a[i + 1] = self.activation.f(self.z[i + 1])
        # z.append(SoftMax.f(np.dot(np.c_[z[-1], np.ones(n)], w[-1])))
        return self.a
