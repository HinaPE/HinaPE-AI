import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, N):
    X = torch.normal(0, 1, (N, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
X1, y1 = synthetic_data(true_w, true_b, 10)


# d2l.set_figsize()
# d2l.plt.scatter(X1[:, 0], y1, 1)
# d2l.plt.show()


def data_iter(batch_size, features, labels):
    N = len(features)
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, N)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(5, X1, y1):
        l = loss(net(X, true_w, true_b), y)
        l.sum().backward()
        sgd([true_w, true_b], lr, 5)
