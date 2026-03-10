import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train = False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)

sample_image, sample_label = train_dataset[0]
print("Dimension of a single data sample:", sample_image.shape)

# def train1epoch(train_loader, optimizer, loss):
#     for data in train_loader:
