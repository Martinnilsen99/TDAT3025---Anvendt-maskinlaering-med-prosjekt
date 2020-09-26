# We're making a model for the XOR operator
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from tabulate import tabulate
import matplotlib.cm as cm


class XOR_operator_model:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
                                [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.W2 = torch.tensor(
            [[random.uniform(-1.0, 1.0)], [random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b1 = torch.tensor(
            [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b2 = torch.tensor(
            [[random.uniform(-1.0, 1.0)]], requires_grad=True)

    # First layer function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = XOR_operator_model()

# Observed/training input and output.
x_train = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)

"""--------------- Optimization ----------------"""
epoch = 100_000
learning_rate = 1

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD(
    [model.b1, model.b2, model.W1, model.W2], lr=learning_rate)
for epoch in range(epoch):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()    # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print(f'W1 = {model.W1}, W2 = {model.W2}, b1 = {model.b1}, b2 = {model.b2}, loss = {model.loss(x_train.reshape(-1, 2), y_train)}')

"""--------------- Visualize ----------------"""

# Visualize result
fig = plt.figure('Oppgave C')
ax = fig.gca(projection='3d')
plt.title('XOR-operator')
# set axes limits, labels and create a table of the XOR
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x_1$", "$x_2$", "$f(x)$"],
          cellLoc="center",
          loc="lower right")


x1 = np.arange(0, 1, 0.02)  # Set x1-values on the grid
x2 = np.arange(0, 1, 0.02)  # Set x2-values on the grid
# Calculate y-axis values
y = np.empty([len(x1), len(x2)], dtype=np.double)
for t in range(len(x1)):
    for r in range(len(x2)):
        y[t, r] = float(model.f(torch.tensor([float(x1[t]), float(x2[r])])))

x1, x2 = np.meshgrid(x1, x2)  # Create meshgrid
surf = ax.plot_wireframe(x1, x2, np.array(y))  # Plots the wireframe

# Scatter/plot the points for f(x1, x2) in x_train
xer = [float(x[0]) for x in x_train]
yer = [float(x[1]) for x in x_train]
ax.scatter(xer, yer, y_train)

float(model.f(torch.tensor([1.0, 0.0])))

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=0, azim=-180)

plt.show()
