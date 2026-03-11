import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from models import SimpleNN
import torch.optim as optim

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


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(model, loader, optimizer, criterion, epochs = 3):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(loader):.4f}")


if __name__ == '__main__':
    print("Starting training")
    train(model, train_loader, optimizer, criterion)
    print("Training finished")