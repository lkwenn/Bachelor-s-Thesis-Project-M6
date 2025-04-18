from torchvision import datasets
import numpy as np


# return protocol:
# return trainx, trainy in numpy arrays or ndarrays with no preprocessing

# Prepare preprocessed mnist dataset
def mnist():
    # Download mnist dataset
    trainset = datasets.MNIST('data/cache', train=True, download=True)
    trainx = trainset.data.numpy().astype(np.uint8)
    trainy = trainset.targets.numpy().astype(np.uint8)

    return trainx, trainy


# Prepare preprocessed cifar10 dataset
def cifar10():
    # Download cifar10 dataset
    trainset = datasets.CIFAR10('data/cache', train=True, download=True)

    trainx = trainset.data.astype(np.uint8)
    # Trainspose the data to fit the shape of pytorch
    trainx = np.transpose(trainx, (0, 3, 1, 2))

    trainy = np.array(trainset.targets).astype(np.uint8)

    return trainx, trainy

# Prepare preprocessed Fashion-MNIST dataset
def fashionmnist():
    # Download Fashion-MNIST dataset
    trainset = datasets.FashionMNIST('data/cache', train=True, download=True)
    trainx = trainset.data.numpy().astype(np.uint8)
    trainy = trainset.targets.numpy().astype(np.uint8)

    return trainx, trainy
