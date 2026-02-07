import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist_01(subsample=False):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    def filter_01(dataset):
        xs, ys = [], []
        for x, y in dataset:
            if y in [0,1]:
                x = x.numpy().reshape(28,28)
                if subsample:
                    x = x[::2, ::2]     #  28×28 → 14×14
                xs.append(x.reshape(-1))
                ys.append(1 if y == 0 else -1)
        return np.array(xs), np.array(ys)

    X_train, y_train = filter_01(train_set)
    X_val, y_val     = filter_01(val_set)
    print("unique labels:", np.unique(y_train))

    return X_train, y_train, X_val, y_val
