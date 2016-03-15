from models import *
from plot import *
import itertools

lower = -1
upper = 1

architecture = (2, 3, 2)
eta = 10
nn = NN(architecture, CrossEntropyCost, Sigmoid, eta)

training_set_size = 100000
batch_size = 1000
training_set = np.array([generate_example() for _ in range(training_set_size)])

cost = list()

dimension_samples = 50
resolution = 10
x = np.arange(-1, 1, 2 / dimension_samples)
y = np.arange(-1, 1, 2 / dimension_samples)
all_combinations = np.array([list(x) for x in itertools.product(x, y)])

graf = list()
draw = 1
if draw:
    for _ in range(len(nn.z[-1]) - 1):
        graf.append(PlotContour(x, y, resolution))

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

plt.show()
u = 0
for i in np.split(training_set, training_set_size / batch_size):
    print("u ==", u)
    u += 1
    cijena = nn.train_on_input_batch(i[:, 0], i[:, 1])
    cost.append(cijena)

    nn.z = nn.forward_pass_batch(all_combinations)
    values = nn.z[-1].reshape([dimension_samples, dimension_samples, 2], order="F")
    for j in range(len(graf)):
        graf[j].draw(values[:, :, j])

        # graf[0].draw(values[:, :, 0]/(values[:, :, 0] + values[:, :, 1]))
        # input()

matplotlib.interactive(False)
plt.figure()
plt.plot(cost)
plt.axis((0, training_set_size / batch_size, min(cost), max(cost)))
plt.show()

nn.z = nn.forward_pass_batch(training_set[:, 0])
predicted = nn.z[-1]
for i, item in enumerate(predicted):
    print("Predicted:", item, " \n   Actual: ", training_set[i, 1, :])
    print("argmax:\n", np.argmax(item), " -- ", np.argmax(training_set[i, 1, :]))

tocno = np.sum(np.argmax(predicted, axis=1) == np.argmax(training_set, axis=1)[:, 1])

print("Postotak:", tocno / len(training_set))

plt.figure()
brojevi = np.arange(lower, upper, 0.01)

plt.plot(brojevi, [fun0(i) for i in brojevi])
plt.plot(brojevi, [fun1(i) for i in brojevi])
plt.axes().set_aspect('equal')
plt.axis((lower, upper, lower, upper))

for i, item in enumerate(training_set):
    if np.argmax(item[1]) == np.argmax(predicted[i]):
        pass
        # plt.scatter(i[0, 0], i[0, 1], c=1)
    else:
        plt.scatter(item[0, 0], item[0, 1], c=6)

plt.show()
