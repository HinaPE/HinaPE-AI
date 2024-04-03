import torch
from matplotlib import pyplot as plt
import matplotlib_inline
import numpy as np
import random


def random_pair_iter(_batch_size, _first, _second):
    assert len(_first) == len(_second)
    num = len(_first)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num)])  # index_select need LongTensor as index
        yield _first.index_select(0, j), _second.index_select(0, j)  # dim: 0 means row, 1 means col;


def set_figsize(_figsize=(10, 7)):
    matplotlib_inline.backend_inline.set_matplotlib_formats()
    plt.rcParams['figure.figsize'] = _figsize


def linear_regression(_X, _w, _b):
    return torch.mm(_X, _w) + _b


def squared_loss(_y_hat, y):
    return (_y_hat - y.view(_y_hat.size())) ** 2 / 2


def create_data_set(_num_inputs, _num_examples):
    _true_w = [2, -3.4]
    _true_b = 4.2
    features = torch.randn(_num_examples, _num_inputs, dtype=torch.float32)
    labels = _true_w[0] * features[:, 0] + _true_w[1] * features[:, 1] + _true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    return [features, labels]


def sgd(_params, _lr, _batch_size):
    for _param in _params:
        _param.data -= _lr * _param.grad / _batch_size


if __name__ == '__main__':
    [features, labels] = create_data_set(2, 1000)
    w = torch.tensor(np.random.normal(0, 0.01, (2, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    w.requires_grad_()
    b.requires_grad_()

    lr = 0.03
    num_epochs = 3
    net = linear_regression
    loss = squared_loss

    batch_size = 10
    for epoch in range(num_epochs):
        for X, y in random_pair_iter(batch_size, features, labels):
            ls = loss(net(X, w, b), y).sum()
            ls.backward()
            sgd([w, b], lr, batch_size)

            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
