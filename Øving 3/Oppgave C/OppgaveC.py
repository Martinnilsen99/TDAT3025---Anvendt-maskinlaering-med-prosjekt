"""
The task given was to improve the model using for example ReLU or dropout
I use the nn.py from our lecturer as a base, and works mostly inside the class, within the init and logits-method
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST(
    './data', train=True, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST(
    './data', train=False, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        # Initialises the Model layers (includes initialized model variables):
        # Could also write stride=2 here, but I suspect it is the standard for kernelsize 2 (as 2x2 will jump 2 to the right for the next 2x2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        # x = self.pool(self.conv1(x))
        # x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.dropout(F.relu(self.conv1(x)), p=0.1))
        
        # x = self.pool(self.conv2(x))
        # x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.dropout(F.relu(self.conv2(x)), p=0.1))
        
        x = self.fc1(x.view(-1, 64 * 7 * 7))
        x = self.fc2(x.view(-1, 1024))
        x = self.fc3(x.view(-1, 100))
        return x

    # Predictor
    def f(self, x):
        return torch.softmax(self.forward(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.forward(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

epoch = 20
learning_rate = 0.001

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i, epoch in enumerate(range(epoch)):
    for batch in range(len(x_train_batches)):
        # Compute loss gradients
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print(f'Accuracy in epoch {i+1}: {model.accuracy(x_test, y_test)*100}%')
