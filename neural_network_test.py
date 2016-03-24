from models import *
from dataset import *

Line1d()
input()
architecture = (2, 2, 2)
eta = 10
nn = NN(architecture, CrossEntropyCost, Sigmoid, eta)

training_set_size = 1000000
batch_size = 1000
tw = TwoLines(training_set_size, n_graphs=1, resolution=20, dimension_samples=60)
cost = list()

div = 5
for i, batch in enumerate(np.split(tw.tr_set, tw.trset_size / batch_size)):
    cijena = nn.train_on_input_batch(batch[:, 0], batch[:, 1])
    cost.append(cijena)

    if i / (tw.trset_size / batch_size) > 0.2:
        div = 50
    if i % div == 0:
        print("Batch #", i, "  Last cost: ", cijena, sep="")
        tw.plot(nn.forward_pass_batch(tw.all_combinations)[-1], cost)
    if i < 3:
        plt.pause(0.0001)

    if cijena < 0.01:
        break

print("Weights:")
for i in nn.weights:
    print(i)
print("------")

last_layer_all = nn.forward_pass_batch(tw.tr_set[:, 0])[-1]

pred = np.argmax(last_layer_all, axis=1)
actual = np.argmax(tw.tr_set[:, 1], axis=1)
tocno = np.sum(pred == actual)

# for i in range(len(pred)):
#     if pred[i] != actual[i]:
#         print("Predicted: ", last_layer_all[i], "\n tw:", tw.tr_set[i], "\n pred: ", pred[i], " actual:", actual[i])

print("Accuracy: ", 100 * tocno / tw.trset_size, "%", sep="")
