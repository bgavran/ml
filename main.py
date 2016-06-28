from models import *
from observers import *
from dataset import *

architecture = (2, 2, 2)
eta = 10
n_batches = 1000
batch_size = 100

nn = NN(architecture, CrossEntropyCost, Sigmoid, eta,
        observers=[ConsoleOutput(), NetworkOutput(n_graphs=1, plot_cost=True)])
tw = Data(TwoLines)

div = 5
for i, (batch_x, batch_y) in enumerate(tw.next_batch(n_batches, batch_size)):
    last_cost = nn.train_on_input_batch(batch_x, batch_y)

    if i / n_batches > 0.2:
        div = 50
    if i % div == 0:
        nn.notify(i, last_cost, tw.data_feed)

    if last_cost < 0.01:
        break

print("Weights:")
for i in nn.weights:
    print(i)
print("------")

much_number = 10000
much_data_x, much_data_y = next(tw.next_batch(1, much_number))
last_layer_all = nn.forward_pass_batch(much_data_x)[-1]

pred = np.argmax(last_layer_all, axis=1)
actual = np.argmax(much_data_y, axis=1)
tocno = np.sum(pred == actual)

print("Accuracy: ", 100 * tocno / much_number, "%", sep="")
