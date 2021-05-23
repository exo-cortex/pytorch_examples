#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
training_data = datasets.EMNIST(
    root="data",
    split="byclass",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
# test_data = datasets.FashionMNIST(
test_data = datasets.EMNIST(
    root="data",
    split="byclass",
    train=False,
    download=True,
    transform=ToTensor(),
)
print(training_data)
print(test_data)